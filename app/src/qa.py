# import torch
import logging
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from transformers import (
	AutoConfig,
	AutoModelForQuestionAnswering,
	AutoTokenizer,
	EvalPrediction,
	HfArgumentParser,
	PreTrainedTokenizerFast,
	TrainingArguments,
	default_data_collator,
	set_seed,
	pipeline
)

logger = logging.getLogger()

class QA_Pipeline():
	def __init__(
		self,
		model: str,
		device: str = "cpu",
		topk: int = 1
	):
		self.device = -1 if device=='cpu' else 0
		self.model = pipeline("question-answering",model=model,device=self.device)
		self.topk = topk

	def span_prediction(
		self,
		questions,
		contexts,
		ids = None
	):
		return self.model(question=questions, context=contexts,topk=self.topk)

class QA_HF():
	def __init__(
		self,
		model: str,
		batch_size: int,
		device: str = "cpu",
		max_seq: int = 384,
		stride: int = 128,
		n_best: int = 1
	):
		self.device = -1 if device=='cpu' else 0
		self.batch_size = batch_size
		self.max_seq = max_seq
		self.stride = stride
		self.n_best = n_best

		config = AutoConfig.from_pretrained(model)
		self.tokenizer = AutoTokenizer.from_pretrained(
			model,
			use_fast=True
		)
		self.model = AutoModelForQuestionAnswering.from_pretrained(
			model,
			config=config
		)
		self.model = self.model.to(device)

	# Validation preprocessing
	def prepare_validation_features(
		self,
		questions,
		contexts,
		ids
		):
		# Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
		# in one example possible giving several features when a context is long, each of those features having a
		# context that overlaps a bit the context of the previous feature.
		tokenized_examples = self.tokenizer(
			questions,
			contexts,
			truncation="only_second",
			max_length=self.max_seq,
			stride=self.stride,
			return_overflowing_tokens=True,
			return_offsets_mapping=True,
			padding="max_length"
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
			tokenized_examples["example_id"].append(ids[sample_index])

			# Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
			# position is part of the context or not.
			tokenized_examples["offset_mapping"][i] = [
				(o if sequence_ids[k] == context_index else None)
				for k, o in enumerate(tokenized_examples["offset_mapping"][i])
			]

		return tokenized_examples

	def span_prediction(
		self,
		ids,
		questions,
		contexts
	):
		context_dataset = self.prepare_validation_features(ids,questions,contexts)

		context_sampler = SequentialSampler(context_dataset)

		logger.info(context_dataset)

		predict_loader = DataLoader(
            context_dataset,
            sampler=context_sampler,
            batch_size=self.batch_size
        )

		for step, inputs in enumerate(predict_loader):
			pass


		# output = 
		return {}




