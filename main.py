from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from graph import workflow

app = FastAPI()
app.mount("/", StaticFiles(directory="static", html=True), name="static")

class Chat(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: Chat):
    resp = workflow.invoke({"messages":[{"role":"user","content":req.message}]})
    return {"reply": resp["messages"][-1].content}
