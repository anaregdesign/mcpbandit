## mcpbandit

An experimental library for routing LLM agents with contextual bandits. `QuestionBasedContextExtractor` builds feature vectors from user input, and policies such as `ThompsonSamplingRegistry` (or `UCBRegistry`) learn which agent to run by balancing exploration and exploitation with every turn.

## Install (uv)

Requires Python 3.13+. Start by cloning the repo, then set up with uv:

```bash
git clone https://github.com/your-org/mcpbandit.git
cd mcpbandit
uv python install 3.13          # Provision Python locally if needed
uv venv                         # Create .venv
source .venv/bin/activate       # On Windows: .venv\\Scripts\\activate
uv sync                         # Install dependencies
```

Examples that call OpenAI models need `OPENAI_API_KEY` set before running.

## Example: agent selection loop

`examples/agents/cli_chat_loop.py` implements a chat loop that picks between two agents for every user turn using a bandit policy. Run it with:

```bash
uv run python examples/agents/cli_chat_loop.py
```

Key pieces (excerpted from the example):

```python
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
        # Report reward for the previous arm so the bandit updates its belief
        registry.observe(
            arm_id=previous_arm.id,
            reward=current_context.feedback,
            context=previous_context.feature_vector,
        )

    # Sample parameters (Thompson Sampling) to pick the next agent for this context
    chosen_arm: Arm[Agent] = registry.select(
        current_context.feature_vector
    )
    result = await Runner.run(
        starting_agent=chosen_arm.body,
        input=user_input,
        session=session,
    )
    print(f"{chosen_arm.body.name}: {result.final_output}\n")

    previous_arm = chosen_arm
    previous_context = current_context
```

What happens:
- `QuestionBasedContextExtractor` asks the LLM three yes/no-ish questions and turns the numeric answers into a feature vector.
- Bandit step (core point): `ThompsonSamplingRegistry` registers each agent as an arm, `observe` feeds back the reward from the previous turn, and `select` samples parameters to choose which agent to run for the current context.
- The chosen agent is executed with `Runner.run`, and the resulting feedback becomes the reward signal for the next `observe`/`select` cycle.

You can swap in different agents, context extractors, or bandit policies to adapt the routing to your use case.
