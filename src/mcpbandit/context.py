from abc import ABC, abstractmethod
from typing import Any

from attr import dataclass
from numpy.typing import NDArray
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field



class Question(BaseModel):
    id: int = Field(description="The unique identifier for the question.")
    question: str = Field(description="The question being asked.")


class Answer(BaseModel):
    id: int = Field(description="The unique identifier for the question.")
    answer: float = Field(
        description="The numeric answer to the question. If it is yes/no, use 1.0/0.0.",
        ge=-1.0,
        le=1.0,
    )


class Context(BaseModel):
    answers: list[Answer] = Field(
        description="List of answers associated with the context."
    )
    feedback: float = Field(
        description="Evaluate sentiment feedback of previous interaction in the range [-1.0, 1.0].",
        ge=-1.0,
        le=1.0,
    )

    @property
    def feature_vector(self) -> NDArray[np.float64]:
        """Convert the context answers to a feature vector."""
        sorted_answers = sorted(self.answers, key=lambda ans: ans.id)
        return np.array([ans.answer for ans in sorted_answers], dtype=np.float64)


class ContextExtractor(ABC):
    """Interface for extracting context features."""

    @property
    @abstractmethod
    def context_length(self) -> int:
        pass

    @abstractmethod
    async def extract(self, input_text: str) -> Context:
        pass


@dataclass
class QuestionBasedContextExtractor(ContextExtractor):
    llm_client: AsyncOpenAI
    model_name: str
    questions: list[Question]
    api_kwargs: dict[str, Any]

    @property
    def context_length(self) -> int:
        return len(self.questions)

    async def extract(self, input_text: str) -> Context:
        response = await self.llm_client.responses.parse(
            model=self.model_name,
            instructions=f"""
            <Instructions>
                <Instruction>
                    You are given a piece of text input. Your task is to answer the following questions based on the content of the input.
                    Provide numeric answers in the range [-1.0, 1.0]. For yes/no questions, use 1.0 for 'yes' and 0.0 for 'no'.
                    Additionally, provide a sentiment feedback score for the overall sentiment of the input text, also in the range [-1.0, 1.0].
                    <Questions>
                        {"".join(f'<Question id="{id}">{q}</Question>' for id, q in enumerate(self.questions, start=1))}
                    </Questions>
                </Instruction>
            </Instructions>
            """,
            input=input_text,
            text_format=Context,
            **self.api_kwargs,
        )
        return response.output_parsed
