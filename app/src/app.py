import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from transformers import pipeline
qa = None

from flask import Flask, current_app, request, jsonify
app = Flask(__name__)



@app.route('/span',methods=['POST'])
def span():
	cfg = current_app.config["config"]

	request_data = request.get_json()
	app.logger.info("%s", request_data)
	contexts = [context['body'] for context in request_data['contexts']]
	questions = [request_data['question']]*len(contexts)


	app.logger.debug("%s", questions)

	span_prediction = qa.span_prediction(questions, contexts)
	app.logger.debug("%s", span_prediction)

	return jsonify(span_prediction)



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
	app.run(host=cfg.server.host, port=cfg.server.port)

	
if __name__ == "__main__":
		main()