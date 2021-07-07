# import torch
import logging
import collections
from typing import Optional, Tuple
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from datasets import Dataset
from transformers import (
	AutoConfig,
	AutoModelForQuestionAnswering,
	AutoTokenizer,
	pipeline,
	default_data_collator
)
from transformers.trainer_pt_utils import nested_concat, nested_detach

logger = logging.getLogger()




class QA_Pipeline:
	def __init__(self, model: str, device: str = "cpu", topk: int = 10):
		self.device = -1 if device == "cpu" else 0
		self.model = pipeline("question-answering", model=model, device=self.device)
		self.topk = topk

	def span_prediction(self, ids, questions, contexts):
		predictions = self.model(question=questions, context=contexts, topk=self.topk, max_answer_len=30)
		
		assert len(predictions) == len(ids) * self.topk

		id_index = 0
		eval_predictions = {}
		for i,example_prediction in enumerate(predictions):
			if i!=0 and i%self.topk == 0:
				id_index+=1
			eval_predictions.setdefault(ids[id_index], []).append(example_prediction)

		return eval_predictions


class QA_HF:
	"""
	Code partially adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
	"""
	def __init__(
		self,
		model: str,
		batch_size: int = 100,
		device: str = "cpu",
		max_seq: int = 384,
		stride: int = 128,
		n_best: int = 1,
	):
		self.device = -1 if device == "cpu" else 0
		self.batch_size = batch_size
		self.max_seq = max_seq
		self.stride = stride
		self.n_best = n_best

		config = AutoConfig.from_pretrained(model)
		self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
		self.model = AutoModelForQuestionAnswering.from_pretrained(model, config=config)
		self.model = self.model.to(device)
		self.model.eval()

	# Validation preprocessing
	def prepare_validation_features(self, examples):
		# Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
		# in one example possible giving several features when a context is long, each of those features having a
		# context that overlaps a bit the context of the previous feature.
		tokenized_examples = self.tokenizer(
			examples["questions"],
			examples["contexts"],
			truncation="only_second",
			max_length=self.max_seq,
			stride=self.stride,
			return_overflowing_tokens=True,
			return_offsets_mapping=True,
			padding="max_length",
		)

		# Since one example might give us several features if it has a long context, we need a map from a feature to
		# its corresponding example. This key gives us just that.
		sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

		# For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
		# corresponding example_id and we will store the offset mappings.
		tokenized_examples["example_id"] = []

		for i in range(len(tokenized_examples["input_ids"])):
			# Grab the sequence corresponding to that example (to know what is the context and what is the question).
			sequence_ids = tokenized_examples.sequence_ids(i)
			context_index = 1

			# One example can give several spans, this is the index of the example containing this span of text.
			sample_index = sample_mapping[i]
			tokenized_examples["example_id"].append(examples["id"][sample_index])

			# Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
			# position is part of the context or not.
			tokenized_examples["offset_mapping"][i] = [
				(o if sequence_ids[k] == context_index else None)
				for k, o in enumerate(tokenized_examples["offset_mapping"][i])
			]

		return tokenized_examples

	def span_prediction(self, ids, questions, contexts):

		examples = Dataset.from_dict({
			'id':ids,
			'questions':questions,
			'contexts':contexts
		})

		features = examples.map(
			self.prepare_validation_features,
			batched=True,
			desc="Running tokenizer on validation dataset",
			remove_columns=examples.column_names,
		)		

		context_sampler = SequentialSampler(features)

		predict_loader = DataLoader(
			features.remove_columns(['offset_mapping', 'example_id']), sampler=context_sampler, batch_size=int(self.batch_size), collate_fn=default_data_collator
		)

		predictions = None

		for step, inputs in enumerate(predict_loader):
			outputs = self.model(**inputs)

			logits = tuple(v for k, v in outputs.items())
			logits = nested_detach(logits)
			predictions =  logits if predictions is None else nested_concat(predictions, logits, padding_index=-100)


		eval_preds = self.postprocess_qa_predictions(
			examples,
			features,
			predictions
		)

		# output =
		return eval_preds

	def postprocess_qa_predictions(
		self,
		examples,
		features,
		predictions: Tuple[np.ndarray, np.ndarray],
		version_2_with_negative: bool = False,
		n_best_size: int = 10,
		max_answer_length: int = 30,
		log_level: Optional[int] = logging.WARNING,
	):
		assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
		all_start_logits, all_end_logits = predictions
		

		assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

		# Build a map example to its corresponding features.
		example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
		features_per_example = collections.defaultdict(list)
		for i, feature in enumerate(features):
			features_per_example[example_id_to_index[feature["example_id"]]].append(i)

		# The dictionaries we have to fill.
		all_predictions = collections.OrderedDict()
		all_nbest_json = collections.OrderedDict()
		if version_2_with_negative:
			scores_diff_json = collections.OrderedDict()

		# Logging.
		logger.setLevel(log_level)
		logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

		# Let's loop over all the examples!
		for example_index, example in enumerate(tqdm(examples)):
			# Those are the indices of the features associated to the current example.
			feature_indices = features_per_example[example_index]

			min_null_prediction = None
			prelim_predictions = []

			# Looping through all the features associated to the current example.
			for feature_index in feature_indices:
				# We grab the predictions of the model for this feature.
				start_logits = all_start_logits[feature_index].numpy()
				end_logits = all_end_logits[feature_index].numpy()
				# This is what will allow us to map some the positions in our logits to span of texts in the original
				# context.
				offset_mapping = features[feature_index]["offset_mapping"]
				# Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
				# available in the current feature.
				token_is_max_context = features[feature_index].get("token_is_max_context", None)

				# Update minimum null prediction.
				feature_null_score = start_logits[0] + end_logits[0]
				if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
					min_null_prediction = {
						"offsets": (0, 0),
						"score": feature_null_score,
					}

				# Go through all possibilities for the `n_best_size` greater start and end logits.
				start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
				end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
				for start_index in start_indexes:
					for end_index in end_indexes:
						# Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
						# to part of the input_ids that are not in the context.
						if (
							start_index >= len(offset_mapping)
							or end_index >= len(offset_mapping)
							or offset_mapping[start_index] is None
							or offset_mapping[end_index] is None
						):
							continue
						# Don't consider answers with a length that is either < 0 or > max_answer_length.
						if end_index < start_index or end_index - start_index + 1 > max_answer_length:
							continue
						# Don't consider answer that don't have the maximum context available (if such information is
						# provided).
						if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
							continue
						prelim_predictions.append(
							{
								"offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
								"score": start_logits[start_index] + end_logits[end_index],
							}
						)
			if version_2_with_negative:
				# Add the minimum null prediction
				prelim_predictions.append(min_null_prediction)
				null_score = min_null_prediction["score"]

			# Only keep the best `n_best_size` predictions.
			predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

			# Add back the minimum null prediction if it was removed because of its low score.
			if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
				predictions.append(min_null_prediction)

			# Use the offsets to gather the answer text in the original context.
			context = example["contexts"]
			for pred in predictions:
				offsets = pred.pop("offsets")
				pred["start"] = offsets[0]
				pred["end"] = offsets[1]
				pred["answer"] = context[offsets[0] : offsets[1]]

			# In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
			# failure.
			if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["answer"] == ""):
				predictions.insert(0, {"answer": "empty", "score": 0.0})

			# Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
			# the LogSumExp trick).
			scores = np.array([pred.pop("score") for pred in predictions])
			exp_scores = np.exp(scores - np.max(scores))
			probs = exp_scores / exp_scores.sum()

			# Include the probabilities in our predictions.
			for prob, pred in zip(probs, predictions):
				pred["score"] = prob

			all_nbest_json[example["id"]] = [
				{k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
				for pred in predictions
			]


		return all_nbest_json
