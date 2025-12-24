## mcpbandit

An experimental library for choosing LLM agents with contextual bandits. `QuestionBasedContextExtractor` builds feature vectors from user input, and policies such as `ThompsonSamplingRegistory` learn which agent to route to as feedback accumulates.

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

`examples/agents/cli_chat_loop.py` implements a chat loop that picks between two agents for every user turn. Run it with:

```bash
uv run python examples/agents/cli_chat_loop.py
```

How it works:
- A shared `AsyncOpenAI` client powers both the context extractor and the agents, which are defined as polite and casual personas.
- `QuestionBasedContextExtractor` asks the LLM three questions (e.g., is this technical? is it Japanese?) and turns the numeric answers into `Context.feature_vector`.
- `ThompsonSamplingRegistory` registers each agent as an arm and, after each reply, `observe` reports the previous interaction’s feedback (`Context.feedback`) to update arm statistics.
- On the next turn `select` samples parameters to choose an arm, and `Runner.run` executes the chosen agent; the resulting context feeds the following learning step.

You can swap in different agent sets, context extractors, or bandit policies to adapt the online selection behavior to your use case.
