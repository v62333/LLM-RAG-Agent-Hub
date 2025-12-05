from fastapi import APIRouter
from app.models.schemas import IngestDocsRequest, IngestDocsResponse
from app.services.ingest_service import ingest_files_to_collection

router = APIRouter()


@router.post("/docs", response_model=IngestDocsResponse)
async def ingest_docs(req: IngestDocsRequest) -> IngestDocsResponse:
    success_count, failed_files = ingest_files_to_collection(
        file_paths=req.file_paths,
        collection_name=req.collection.value,
    )
    return IngestDocsResponse(
        success_count=success_count,
        failed_files=failed_files,
    )
