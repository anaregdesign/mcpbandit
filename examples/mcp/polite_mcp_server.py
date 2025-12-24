from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name=__name__,
    host="localhost",
    port=8001,
)


@mcp.prompt(
    name="polite_mcp_prompt",
    description="丁寧な表現を使い礼儀正しく対話するプロンプトです",
)
def polite_mcp_prompt(text: str):
    return (
        "あなたは丁寧で礼儀正しいエージェントです\n"
        "常に敬語を使い、相手に配慮した回答を心がけてください\n"
        "感情よりも論理的な正しさを重視してください\n"
        "---\n"
        f"{text}"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
