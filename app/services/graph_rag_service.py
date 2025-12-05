from app.models.schemas import RagAnswer
from .rag_service import retrieve_top_k, build_context, build_prompt_with_context
from app.llm.llm_client import get_llm_client


async def answer_with_graph_rag(
    question: str,
    collection: str,
    top_k: int = 5,
) -> RagAnswer:
    client = get_llm_client()
    chunks = retrieve_top_k(question, collection, top_k=top_k)
    context = build_context(chunks)
    prompt = (
        "你是一位能利用文件結構與關聯的 GraphRAG 助手。\n"
        + build_prompt_with_context(question, context)
    )

    llm_result = await client.generate(
        system_prompt=None,
        user_prompt=prompt,
        temperature=0.2,
        max_tokens=512,
    )

    return RagAnswer(
        answer=llm_result["output"],
        strategy="graph_rag",
        sources=chunks,
        metadata={"model": llm_result["model"]},
    )
