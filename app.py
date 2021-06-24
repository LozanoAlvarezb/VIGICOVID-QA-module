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
	question = [request_data['question']]*len(contexts)


	app.logger.debug("%s", question)

	span_prediction = qa(question=question, context=contexts,topk=cfg.qa.topk)
	app.logger.debug("%s", span_prediction)

	return jsonify(span_prediction)



@hydra.main(config_path="conf", config_name="server")
def main(cfg: DictConfig):
	
	global qa
	qa = pipeline("question-answering",model=cfg.qa.model)
								# ,
								# device=cfg.qa.device)

	app.config.from_mapping(cfg.flask)
	with open_dict(cfg):
			del cfg["flask"]

	app.config["config"] = cfg
	app.run(host=cfg.server.host, port=cfg.server.port)

	
if __name__ == "__main__":
		main()