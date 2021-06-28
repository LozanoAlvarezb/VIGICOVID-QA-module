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
	ids = [context['id'] for context in request_data['contexts']]
	scores = [context['score'] for context in request_data['contexts']]
	contexts = [context['text'] for context in request_data['contexts']]
	questions = [request_data['question']]*len(contexts)


	# app.logger.debug("%s", questions)

	try:
		span_prediction = qa.span_prediction(questions, contexts,ids)
		app.logger.debug("%s", span_prediction)

		results = [{
			"id": id,
			"score": score,
			"span": [span['start'],span['end']]
		} for id,score,span in zip(ids,scores,span_prediction)]

		response = {
			"error": None,
			"results": sorted(results, key=lambda x: x["score"], reverse=True)
		}

	except:
		response = {
			"error": "Error",
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
	with open_dict(cfg):
		del cfg["flask"]

	app.config["config"] = cfg
	http_server = WSGIServer((cfg.server.host, cfg.server.port), app) 	
	# app.run(host=cfg.server.host, port=cfg.server.port)
	http_server.serve_forever()    


if __name__ == "__main__":
	main()
