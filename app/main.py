from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

# Set up paths using centralized utility
# Import directly from project root (app/ is in project root, src/ is sibling)
from src.utils.path_setup import setup_paths
setup_paths(include_app=True)

from api.routes import router

app = FastAPI(title="Nowcasting API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")

# Include API routes
app.include_router(router, prefix="/api", tags=["api"])

# Root endpoint - serve index.html
@app.get("/")
async def read_root():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))

# Agent functionality removed - not used in simple app

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    from utils import OUTPUTS_DIR, CONFIG_DIR
    
    # Ensure directories exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Services are initialized as singletons in api/dependencies.py
