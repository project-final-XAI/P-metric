from abc import ABC

from evaluation.judging.base import JudgingModel


class LLMJudgeInterface(JudgingModel, ABC):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    