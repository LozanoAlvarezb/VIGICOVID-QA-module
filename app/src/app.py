import hydra
import numpy as np
import os, sys
from omegaconf import DictConfig, OmegaConf, open_dict
from collections import defaultdict
from gevent.pywsgi import WSGIServer

# Make this the current work dir
abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)

import utils

from flask import Flask, current_app, request, jsonify
app = Flask(__name__)

qa_module = None


@app.route('/span',methods=['POST'])
def span():
	cfg = current_app.config["config"]

	request_data = request.get_json()
	app.logger.debug("%s", request_data)

	# number of answer for each document
	qa_cut = request_data["qa_cut"] if "qa_cut" in request_data else int(cfg['qa_cut'])
	sim_threshold = float(cfg['sim_threshold'])

	app.logger.info("Processing %d questions",len(request_data))

	questions , contexts, ir_scores, ids = utils.get_data(request_data)

	try:
		span_prediction = qa_module.span_prediction(ids, questions, contexts)

		unsorted_results = utils.get_unsorted_results(span_prediction,ir_scores,contexts,request_data,qa_cut)

		results = []
		for question in unsorted_results:

			#Check for similar answers
			sorted_results = sorted(unsorted_results[question], key=lambda x: x["score"], reverse=True)
			final_results=[sorted_results[0]]
			for answer in sorted_results[1:]:
				if any([utils.jaccard_sim(answer['text'],ranked['text']) >= sim_threshold for ranked in final_results]):
					continue
				final_results.append(answer)

			results.append({
				"question": question,
				"spans": final_results
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
