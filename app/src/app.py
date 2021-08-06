import hydra
import os, sys
from omegaconf import DictConfig, OmegaConf, open_dict
from collections import defaultdict
from gevent.pywsgi import WSGIServer

import qa

# Make this the current work dir
abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)

from flask import Flask, current_app, request, jsonify
app = Flask(__name__)

qa_module = None


@app.route('/span',methods=['POST'])
def span():
	cfg = current_app.config["config"]

	request_data = request.get_json()
	app.logger.debug("%s", request_data)

	# number of answer for each document
	# qa_cut = int(request.args.get('qa_cut'))
	qa_cut = int(cfg['qa_cut'])

	app.logger.info("Processing %d questions",len(request_data))

	questions = [padded_question for question in request_data for padded_question in [question['question']]*len(question['contexts']) ]
	ir_scores = [context['score'] for question in request_data for context in question['contexts']]
	contexts = [context['text'] for question in request_data for context in question['contexts']]
	# Combine id with question to cover the case where the same document id is retrieved for different questions
	ids = [f"{i}-{context['id']}" for i,question in enumerate(request_data) for context in question['contexts']]
	
	try:
		span_prediction = qa_module.span_prediction(ids, questions, contexts)
		# app.logger.debug("%s", span_prediction)

		sorted_results = defaultdict(list)
		for q_id, ir_score in zip(span_prediction, ir_scores):
			q_index, id = q_id.split("-", maxsplit=1)
			question = request_data[int(q_index)]['question']
			id_scores = []
			score_lambda = lambda x: (ir_score + x['score']) /2
			# avoid duplicates: unique by score, similar spans should
			# yield similar scores. If span_predictions were already sorted
			# by score, no need to sort here nor later. ir_score by questions
			# is a fixed quantity, won't change result
			for answer in sorted(span_prediction[q_id], key=score_lambda, reverse=True):
				score = score_lambda(answer)
				if score not in id_scores and len(id_scores) <= qa_cut:
					sorted_results[question].append({
						"id": id,
						"text": answer['answer'],
						"score": score,
						"span": [answer['start'], answer['end']]
					})
					id_scores.append(score)

				if len(id_scores) == qa_cut:
					break

		# for id,ir_score,span,question in zip(ids,ir_ir_scores,span_prediction,questions):
		# 	sorted_results.setdefault(question, []).append({
		# 	"id": id,
		# 	"text": span['answer'],
		# 	"score": (ir_score+span['score'])/2,
		# 	"span": [span['start'],span['end']]
		# })

		results = []
		for question in sorted_results:
			results.append({
				"question": question,
				"spans": sorted_results[question]
			})

		response = {
			"error": None,
			"results": results
		}

	except Exception as e:
		response = {
			"error": f"Unexpected error:{e}",
			"results": None
		}
		app.logger.critical(e, exc_info=True)
	return jsonify(response)




@hydra.main(config_path="../conf", config_name="app")
def main(cfg: DictConfig):
	app.logger.info("CFG:")
	app.logger.info("%s", OmegaConf.to_yaml(cfg))
	
	global qa_module
	qa_module = hydra.utils.instantiate(cfg.qa_modules[cfg.qa])

	app.config.from_mapping(cfg.flask)
	# with open_dict(cfg):
	# 	del cfg["flask"]

	app.config["config"] = cfg
	if cfg.flask.ENV == 'production':
		http_server = WSGIServer((cfg.server.host, cfg.server.port), app) 	
		http_server.serve_forever()   
	# debug mode
	else:
		app.run(host=cfg.server.host, port=cfg.server.port)
	 


if __name__ == "__main__":
	main()
