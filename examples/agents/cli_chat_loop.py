import asyncio

from agents import Agent, ModelSettings, Runner, SQLiteSession
from openai import AsyncOpenAI

from mcpbandit.bandit import Arm, ThompsonSamplingRegistry
from mcpbandit.context import Context, QuestionBasedContextExtractor

async def main() -> None:
    # Shared LLM client for both the context extractor and agents
    llm_client = AsyncOpenAI()

    # Two simple agent personas with different instruction styles
    polite_agent = Agent(
        name="polite_agent",
        instructions="You are a polite and helpful assistant. Always respond courteously.",
        model="gpt-5.1",
        model_settings=ModelSettings(
            reasoning={"effort": "none"}
        ),
    )

    casual_agent = Agent(
        name="casual_agent",
        instructions="You are a casual and friendly assistant. Use informal language.",
        model="gpt-5.1",
        model_settings=ModelSettings(
            reasoning={"effort": "none"}
        ),
    )

    # In-memory session keeps the conversation state for the runner
    session = SQLiteSession(":memory:")

    # Extract a compact feature vector from the latest user input via LLM Q&A
    context_extractor = QuestionBasedContextExtractor(
        llm_client=llm_client,
        model_name="gpt-5.1",
        questions=[
            "Is the user asking a question?",
            "Is the topic technical?",
            "Is the input in Japanese?",
        ],
        api_kwargs={"reasoning": {"effort": "none"}},
    )

    # Bandit registry that chooses an agent based on the extracted context
    registry = ThompsonSamplingRegistry(
        context_length=context_extractor.context_length
    )
    registry.add(polite_agent)
    registry.add(casual_agent)

    previous_arm: Arm[Agent] | None = None
    previous_context: Context | None = None

    while True:
        user_input = input("user: ")

        current_context = await context_extractor.extract(user_input)
        if previous_arm and previous_context:
            # Report reward for the previous choice so the bandit can learn
            registry.observe(
                arm_id=previous_arm.id,
                reward=current_context.feedback,
                context=previous_context.feature_vector,
            )
        
        # Select the next agent using Thompson Sampling over contexts
        chosen_arm: Arm[Agent] = registry.select(
            current_context.feature_vector
        )

        # Run the selected agent and print its final reply
        result = await Runner.run(
            starting_agent=chosen_arm.body,
            input=user_input,
            session=session,
        )
        print(f"{chosen_arm.body.name}: {result.final_output}\n")

        previous_arm = chosen_arm
        previous_context = current_context


if __name__ == "__main__":
    asyncio.run(main())
