# import torch
import logging
from transformers import pipeline

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
        contexts
    ):
        return self.model(question=questions, context=contexts,topk=self.topk)

