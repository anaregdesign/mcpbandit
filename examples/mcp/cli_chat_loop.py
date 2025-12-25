import asyncio

import fastmcp
from openai import AsyncOpenAI

from mcpbandit.bandit import Arm, BanditRegistry, ThompsonSamplingRegistry
from mcpbandit.context import Context, QuestionBasedContextExtractor


async def main() -> None:
    llm = AsyncOpenAI()
    extractor = QuestionBasedContextExtractor(
        llm_client=llm,
        model_name="gpt-5.1",
        questions=[
            "これは質問ですか？",
            "専門的な技術の話題ですか？",
            "日本語ですか？",
        ],
        api_kwargs={"reasoning": {"effort": "none"}},
    )
    registry: BanditRegistry[fastmcp.Client] = ThompsonSamplingRegistry(
        context_length=extractor.context_length,
        alpha=0.5,
    )
    registry.add(fastmcp.Client("http://localhost:8000/mcp"))
    registry.add(fastmcp.Client("http://localhost:8001/mcp"))

    previous_arm: Arm[fastmcp.Client] | None = None
    previous_context: Context | None = None

    while True:
        user_input = input("ユーザー: ")

        # Generate current context and observe previous feedback
        current_context = await extractor.extract(user_input)
        if previous_arm and previous_context:
            registry.observe(
                arm_id=previous_arm.id,
                reward=current_context.feedback,
                context=previous_context.feature_vector,
            )

        chosen_arm: Arm[fastmcp.Client] = registry.select(
            current_context.feature_vector
        )
        async with chosen_arm.body as client:
            # この中でMCPを使った処理を行う
            prompt_list = await client.list_prompts()
            get_prompt_result = await client.get_prompt(
                prompt_list[0].name, {"text": user_input}
            )

            response = await llm.responses.create(
                model="gpt-5.1",
                input=get_prompt_result.messages[0].content.text,
                reasoning={"effort": "none"},
            )

            print(
                "---\n"
                f"エージェント({response.output_text}\n"
            )

        previous_arm = chosen_arm
        previous_context = current_context


if __name__ == "__main__":
    asyncio.run(main())
