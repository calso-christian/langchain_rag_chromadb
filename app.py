from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from lc_conversational_helper import chat
from lc_question_helper import process_query
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS (required if calling from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to allowed domains for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class AskRequest(BaseModel):
    message: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

# Chatbot API (Conversational)
@app.post('/chat')
async def chat_bot(request: ChatRequest):
    try:
        logger.info(f"Chat request from user {request.user_id}: {request.message}")
        response = chat(request.user_id, request.message)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat_bot: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Question Answering API
@app.post('/ask')
async def ask_bot(question: AskRequest):
    try:
        logger.info(f"Ask request: {question.message}")
        response = process_query({"query": question.message})  # ✅ Fix: Convert input
        return {"result": response}
    except Exception as e:
        logger.error(f"Error in ask_bot: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Chatbot API is running!"}
