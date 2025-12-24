from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name=__name__,
    host="localhost",
    port=8000,
)


@mcp.prompt(
    name="casual_mcp_prompt",
    description="砕けた表現を使い共感をベースに対話するプロンプトです",
)
def casual_mcp_prompt(text: str):
    return (
        "あなたは相談者の友達として接するエージェントです\n"
        "論理よりも感情に寄り添う形で回答してください\n"
        "敬語よりも砕けた表現を使い、フレンドリーな口調で話してください\n"
        "---\n"
        f"{text}"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
