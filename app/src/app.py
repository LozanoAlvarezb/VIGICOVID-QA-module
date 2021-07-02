import hydra
import os, sys
from omegaconf import DictConfig, OmegaConf, open_dict
from gevent.pywsgi import WSGIServer

import qa

# Make this the current work dir
abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)

from flask import Flask, current_app, request, jsonify
app = Flask(__name__)

qa = None


@app.route('/span',methods=['POST'])
def span():
	# cfg = current_app.config["config"]

	request_data = request.get_json()
	app.logger.debug("%s", request_data)


	ids = [context['id'] for question in request_data for context in question['contexts']]
	scores = [context['score'] for question in request_data for context in question['contexts']]
	contexts = [context['text'] for question in request_data for context in question['contexts']]
	questions = [padded_question for question in request_data for padded_question in [question['question']]*len(question['contexts']) ]

	# app.logger.debug("%s", questions)

	try:
		span_prediction = qa.span_prediction(questions, contexts,ids)
		app.logger.debug("%s", span_prediction)

		unsorted_results = {}
		for id,score,span,question in zip(ids,scores,span_prediction,questions):
			unsorted_results.setdefault(question, []).append({
			"id": id,
			"score": score,
			"span": [span['start'],span['end']]
		})

		results = []
		for question in unsorted_results:
			results.append({
				"quesiton": question,
				"spans": sorted(unsorted_results[question], key=lambda x: x["score"], reverse=True)
			})
		response = {
			"error": None,
			"results": results
		}

	except :
		response = {
			"error": f"Unexpected error:{sys.exc_info()[0]}",
			"results": None
		}
	return jsonify(response)




@hydra.main(config_path="../conf", config_name="app")
def main(cfg: DictConfig):
	app.logger.info("CFG:")
	app.logger.info("%s", OmegaConf.to_yaml(cfg))
	
	global qa
	qa = hydra.utils.instantiate(cfg.qa_modules[cfg.qa])

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
