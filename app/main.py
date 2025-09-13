from fastapi import FastAPI
from app.chatbot import Chatbot
from app.schemas import ChatRequest, ChatResponse

app = FastAPI(title="Simple Chatbot API")
bot = Chatbot()

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response = bot.chat(req.session_id, req.user_input)
    return ChatResponse(session_id=req.session_id, response=response)

@app.get("/health")
def health():
    return {"status": "ok"}
