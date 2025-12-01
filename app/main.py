from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from api.routes import router
from api.websocket import websocket_endpoint
from agent.graph import workflow

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

# WebSocket endpoint
@app.websocket("/ws/train/{job_id}")
async def websocket_train(websocket: WebSocket, job_id: str):
    await websocket_endpoint(websocket, job_id)

# Root endpoint - serve index.html
@app.get("/")
async def read_root():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))

# Agent chat endpoint (placeholder until agent is fully implemented)
from pydantic import BaseModel

class Chat(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: Chat):
    if workflow is None:
        return {"reply": "Agent not yet implemented"}
    try:
        resp = workflow.invoke({"messages": [{"role": "user", "content": req.message}]})
        return {"reply": resp["messages"][-1].content}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    from services import ModelRegistry, ConfigManager
    from utils import OUTPUTS_DIR, CONFIG_DIR
    
    # Ensure directories exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize registry
    ModelRegistry()
