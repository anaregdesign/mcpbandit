import pytest
import pytest_asyncio
import numpy as np
from openai import AsyncOpenAI

from mcpbandit.context import Context, QuestionBasedContextExtractor


@pytest.fixture
def extractor() -> QuestionBasedContextExtractor:
    client = AsyncOpenAI()
    questions = [
        "Is this a question?",
        "Is the user asking you to tell a joke?",
        "Is this a technical discussion?",
        "Is the user asking for an opinion?",
        "Is the user asking to generate something?",
    ]
    return QuestionBasedContextExtractor(
        llm_client=client,
        model_name="gpt-5.1",
        questions=questions,
        api_kwargs={"reasoning": {"effort": "none"}},
    )


@pytest_asyncio.fixture
async def context(extractor: QuestionBasedContextExtractor) -> Context:
    input_text = (
        "This is a casual, non-technical question asking you to generate a computer "
        "joke. It is not a technical discussion and not asking for your opinion; "
        "please just create something funny."
    )
    return await extractor.extract(input_text)


@pytest.mark.asyncio
async def test_feature_vector_orders_answers_by_id(context: Context) -> None:
    sorted_answers = sorted(context.answers, key=lambda ans: ans.id)
    expected_vector = np.array([ans.answer for ans in sorted_answers], dtype=np.float64)
    assert np.array_equal(context.feature_vector, expected_vector)
    assert context.feature_vector.shape == (len(sorted_answers),)


@pytest.mark.asyncio
async def test_question_based_extractor_answers_correctly(
    context: Context,
) -> None:
    expected_yes = {1, 2, 5}
    expected_no = {3, 4}

    for ans in context.answers:
        if ans.id in expected_yes:
            assert ans.answer > 0.5, f"Question {ans.id} should be yes-ish"
        if ans.id in expected_no:
            assert ans.answer < 0.5, f"Question {ans.id} should be no-ish"

    assert -1.0 <= context.feedback <= 1.0
