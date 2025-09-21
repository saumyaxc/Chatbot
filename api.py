# api.py — Cache index in memory for much faster responses
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rag_chat

app = FastAPI(title="Docs-only Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CACHE THE INDEX IN MEMORY - This is the key optimization!
_cached_index = None
_cached_version = None

def get_index():
    """Load index once and cache it in memory"""
    global _cached_index, _cached_version
    
    current_version = rag_chat.index_version()
    
    # Only reload if version changed or first time
    if _cached_index is None or _cached_version != current_version:
        print(f"Loading index (version: {current_version})")
        _cached_index = rag_chat.load_index()
        _cached_version = current_version
    
    return _cached_index

class QueryIn(BaseModel):
    question: str
    prev_chips: Optional[List[str]] = None
    last_answer: Optional[str] = None

class AnswerOut(BaseModel):
    answer: str
    chips: List[str]
    version: str

@app.get("/health")
def health():
    try:
        get_index()  # Use cached version
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/query", response_model=AnswerOut)
def query(q: QueryIn):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    try:
        # Pass cached index to avoid disk I/O
        embs, meta = get_index()
        answer, chips = rag_chat.answer_with_index(
            embs, meta,
            q.question,
            prev_chips=q.prev_chips or [],
            last_answer=q.last_answer or ""
        )
        return {"answer": answer, "chips": chips, "version": _cached_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
def reindex():
    global _cached_index, _cached_version
    import build_index
    build_index.main()
    # Clear cache so next request loads fresh index
    _cached_index = None
    _cached_version = None
    return {"ok": True}

@app.get("/index_version")
def index_version_route():
    try:
        return {"version": rag_chat.index_version()}
    except Exception as e:
        return {"version": f"unknown-{str(e)}"}