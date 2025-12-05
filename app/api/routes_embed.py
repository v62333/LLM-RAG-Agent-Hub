from fastapi import APIRouter
from app.models.schemas import EmbedRequest, EmbedResponse, EmbedResult
from app.embeddings.embedding_client import get_embedding_client
from app.vectorstore.milvus_client import get_milvus_client

router = APIRouter()


@router.post("/", response_model=EmbedResponse)
async def embed_api(req: EmbedRequest) -> EmbedResponse:
    emb_client = get_embedding_client()
    milvus_client = get_milvus_client()

    vectors = emb_client.embed_texts(req.texts)
    ids = []

    if req.store:
        ids = milvus_client.insert_vectors(
            collection=req.collection.value,
            vectors=vectors,
            metadatas=[{"text": t} for t in req.texts],
        )
    else:
        ids = [None] * len(req.texts)

    results = [
        EmbedResult(text=text, vector_id=vec_id)
        for text, vec_id in zip(req.texts, ids)
    ]
    return EmbedResponse(results=results)
