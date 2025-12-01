"""File upload API endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from api.error_handlers import handle_exceptions
from utils import DATA_DIR
from api.schemas import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
@handle_exceptions
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV data file."""
    # Validate file extension
    if file.filename is None or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    # Save file
    file_path = DATA_DIR / file.filename
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    
    return UploadResponse(
        filename=file.filename,
        path=str(file_path),
        message="File uploaded successfully"
    )

