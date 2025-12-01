"""WebSocket handler for training progress streaming."""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import json
import asyncio

from api.dependencies import training_manager


async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for streaming training progress."""
    await websocket.accept()
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "message": "Connected to training progress stream"
        })
        
        # Poll for updates
        last_progress = -1
        while True:
            status = training_manager.get_status(job_id)
            
            if status is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Job not found"
                })
                break
            
            # Send update if progress changed
            if status["progress"] != last_progress:
                await websocket.send_json({
                    "type": "progress",
                    "data": {
                        "status": status["status"],
                        "progress": status["progress"],
                        "message": status["message"],
                        "error": status.get("error")
                    }
                })
                last_progress = status["progress"]
            
            # Break if completed or failed
            if status["status"] in ["completed", "failed"]:
                await websocket.send_json({
                    "type": "complete",
                    "data": {
                        "status": status["status"],
                        "message": status["message"],
                        "error": status.get("error")
                    }
                })
                break
            
            # Wait before next poll
            await asyncio.sleep(0.5)
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass

