from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from utils import get_answers

app = FastAPI()

class Input(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run_query(data: Input, request: Request):
    result = get_answers(data.documents, data.questions)
    return {"answers": result}
