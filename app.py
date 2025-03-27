from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lc_conversational_helper import chat

app = FastAPI()


class ChatRequest(BaseModel):
    user_id:str
    message:str


@app.post('/chat')
async def chat_bot(request:ChatRequest):
    try:
        response=chat(request.user_id, request.message)
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    return {"message": "Chatbot API is running!"}