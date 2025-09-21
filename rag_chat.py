# rag_chat.py — Accept pre-loaded index to avoid disk I/O
import os, json, re, numpy as np, hashlib, time
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INDEX_DIR = "index"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ------------- helpers -------------
_GREETINGS = {"hi", "hello", "hey", "hi!", "hello!", "hey!", "good morning", "good afternoon", "good evening"}
_FAREWELLS = {"bye", "goodbye", "see you", "thanks", "thank you", "that's all", "done", "exit", "quit"}

def is_greeting(q: str) -> bool:
    return q.strip().lower() in _GREETINGS

def is_farewell(q: str) -> bool:
    qn = q.strip().lower()
    return any(farewell in qn for farewell in _FAREWELLS)

def is_topic_change(q: str, last_answer: str = "") -> bool:
    """Detect if user is changing topics or ending conversation"""
    qn = q.strip().lower()
    
    # Clear topic changes
    if is_greeting(qn) or is_farewell(qn):
        return True
    
    # Very short, generic queries that likely aren't follow-ups
    if len(qn.split()) <= 2 and qn not in {"what", "how", "why", "when", "where"}:
        return True
        
    # User is asking something completely unrelated (no shared keywords)
    if last_answer:
        answer_words = set(last_answer.lower().split())
        query_words = set(qn.split())
        # If no meaningful word overlap, likely a topic change
        common_words = answer_words.intersection(query_words)
        # Filter out very common words
        meaningful_common = [w for w in common_words if len(w) > 3 and w not in {"that", "this", "with", "have", "from", "they", "were", "been", "said", "what", "how", "why", "when", "where"}]
        if len(meaningful_common) == 0 and len(qn.split()) > 1:
            return True
    
    return False

def is_broad_query(q: str) -> bool:
    qn = q.strip().lower()
    return len(qn.split()) <= 2 or any(h in qn for h in ("overview", "tell me about", "general", "summary", "summarize"))

def strip_md(s: str) -> str:
    if not s: return s
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = s.replace("**", "").replace("`", "")
    return s

def is_refusal(msg: str) -> bool:
    m = (msg or "").lower()
    for p in [
        "i don't have that information", "i don't have that information",
        "not in the provided documents", "not in the context",
        "i couldn't find", "i couldn't find", "i'm not sure", "i am not sure",
    ]:
        if p in m: return True
    return False

# ------------- index / embeddings -------------
def load_index():
    """Load embeddings and metadata from disk - should be cached at API level"""
    embs = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return embs, meta

def embed_query(q: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    e = np.array(e, dtype=np.float32)
    e = e / (np.linalg.norm(e) + 1e-12)
    return e

def centroid_vector(embs: np.ndarray) -> np.ndarray:
    c = np.mean(embs, axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    return c

# ------------- retrieval (MMR) -------------
def mmr_select(similarities: np.ndarray, embeddings: np.ndarray, top_k: int, lambda_mult: float = 0.5, pool_size: int = 40):
    pool_idx = np.argsort(-similarities)[:min(pool_size, len(similarities))]
    if len(pool_idx) == 0:
        return []
    selected = [pool_idx[0]]
    selected_set = {pool_idx[0]}
    while len(selected) < min(top_k, len(pool_idx)):
        remaining = [i for i in pool_idx if i not in selected_set]
        best_i, best_score = None, -1e9
        for i in remaining:
            rel = similarities[i]
            div = max(float(embeddings[i] @ embeddings[j]) for j in selected) if selected else 0.0
            score = lambda_mult * rel - (1 - lambda_mult) * div
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i); selected_set.add(best_i)
    return selected

def search(embs, meta, query: Optional[str] = None, *, qvec: Optional[np.ndarray] = None, top_k=6, lambda_mult=0.5, pool_size=40):
    if qvec is None:
        qvec = embed_query(query or "")
    sims = embs @ qvec
    idx = mmr_select(sims, embs, top_k=top_k, lambda_mult=lambda_mult, pool_size=pool_size)
    results = [{"score": float(sims[i]), "source": meta[i]["source"], "text": meta[i]["text"]} for i in idx]
    if results:
        max_sim = results[0]["score"]
        min_score_abs = 0.16
        results = [r for r in results if r["score"] >= max(min_score_abs, 0.80 * max_sim)]
    return results

def build_context(results: List[Dict], *, char_budget: int = 4000, per_chunk_limit: int = 500) -> Tuple[str, List[Dict]]:
    """Keep context small for speed"""
    blocks, taken, total = [], [], 0
    for r in results:
        chunk = strip_md((r["text"] or "").strip())
        if per_chunk_limit and len(chunk) > per_chunk_limit:
            chunk = chunk[:per_chunk_limit]
        add = len(chunk) + 4
        if total + add > char_budget:
            break
        blocks.append(chunk); taken.append(r); total += add
    return "\n\n---\n\n".join(blocks), taken

# ------------- single-call answer + chips -------------
SYSTEM_SINGLE = (
    "You are a helpful assistant grounded strictly in the provided CONTEXT.\n"
    "- Output PLAIN TEXT for the 'answer' (no markdown, no **bold**, no code blocks).\n"
    "- Also propose 3–5 short follow-up questions 'chips' that build on the QUESTION and the ANSWER.\n"
    "- Chips must be answerable strictly from CONTEXT; 6–12 words; avoid duplicates.\n"
    "- If the user says goodbye/farewell (bye, goodbye, thanks, etc.), give a brief friendly farewell and offer general help chips.\n"
    "- If the user greeting is detected (e.g., 'hi'), write a friendly single-sentence greeting first *in the answer*, "
    "then a concise line about how you can help based on CONTEXT. Do this only on greetings.\n"
    "- If unknown, say it plainly in the answer and still offer general chips.\n"
    "- Focus ONLY on the current QUESTION. Do not reference or continue previous topics unless the question directly relates to them.\n"
    "Return a single JSON object with keys: answer (string), chips (array of strings)."
)

USER_SINGLE_TEMPLATE = (
    "QUESTION:\n{q}\n\n"
    "CONTEXT (use this only):\n{ctx}\n\n"
    "Respond ONLY with JSON like:\n"
    "{{\"answer\":\"...plain text...\",\"chips\":[\"chip1\",\"chip2\",\"chip3\"]}}"
)

USER_SINGLE_TEMPLATE_WITH_CONTEXT = (
    "QUESTION:\n{q}\n\n"
    "PREVIOUS_CONTEXT (for follow-up questions only):\n{last}\n\n"
    "CONTEXT (use this only):\n{ctx}\n\n"
    "Respond ONLY with JSON like:\n"
    "{{\"answer\":\"...plain text...\",\"chips\":[\"chip1\",\"chip2\",\"chip3\"]}}"
)

def parse_json_object(text: str):
    """Extract first JSON object from text robustly."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        blob2 = re.sub(r"[^\x20-\x7E]+", " ", blob)
        try:
            return json.loads(blob2)
        except Exception:
            return None

# ------------- main entry - NEW: accept pre-loaded index -------------
def answer_with_index(embs, meta, query: str, top_k=6, prev_chips: Optional[List[str]] = None, last_answer: str = ""):
    """
    Main answer function that accepts pre-loaded embeddings and metadata.
    This avoids disk I/O on every request.
    """
    greeting = is_greeting(query)
    farewell = is_farewell(query)
    broad = is_broad_query(query)
    topic_changed = is_topic_change(query, last_answer)

    # Optimize retrieval for different query types
    if greeting or broad or farewell:
        qvec = centroid_vector(embs) if greeting or farewell or len(query.strip().split()) <= 2 else embed_query(query)
        tk, pool = max(top_k, 6), 30  # smaller pool for speed
        hits = search(embs, meta, qvec=qvec, top_k=tk, lambda_mult=0.55, pool_size=pool)
        ctx_budget = 3500  # smaller context for speed
    else:
        tk, pool = max(top_k, 6), 40
        hits = search(embs, meta, query=query, top_k=tk, lambda_mult=0.5, pool_size=pool)
        ctx_budget = 4000

    if not hits:
        fallback_answer = "I don't have that information in the provided documents."
        fallback_chips = ["Show key topics covered", "Outline main sections in these docs", "What should I read first"]
        return fallback_answer, fallback_chips

    context, strong = build_context(hits, char_budget=ctx_budget, per_chunk_limit=480)

    # Choose the right prompt template based on whether this is a topic change
    if topic_changed or not last_answer.strip() or farewell or greeting:
        # Fresh conversation or topic change - don't include previous context
        prompt = USER_SINGLE_TEMPLATE.format(q=query, ctx=context)
    else:
        # Likely a follow-up question - include minimal previous context
        brief_last = strip_md(last_answer[:200] + "..." if len(last_answer) > 200 else last_answer)
        prompt = USER_SINGLE_TEMPLATE_WITH_CONTEXT.format(q=query, last=brief_last, ctx=context)

    # Single LLM call - add max_tokens to limit response time
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM_SINGLE},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800  # Limit response length for faster generation
    )
    raw = resp.choices[0].message.content.strip()
    data = parse_json_object(raw) or {}

    # sanitize outputs
    ans = strip_md((data.get("answer") or "").strip())
    chips = [strip_md((c or "").strip()) for c in (data.get("chips") or []) if (c or "").strip()]

    # Special handling for farewells - ensure simple goodbye
    if farewell:
        if not any(word in ans.lower() for word in ["goodbye", "bye", "farewell", "thanks"]):
            ans = "Goodbye! Feel free to ask if you need help with anything else."
        # Provide general chips for farewells
        chips = [
            "What topics are covered here",
            "Show me key information",
            "Help me get started"
        ][:3]
    elif len(chips) < 3:
        # Guarantee chips for other cases
        backfill = [
            "Show key topics covered",
            "Outline main sections in these docs", 
            "What should I read first",
            "Give me a quick start checklist",
            "Summarize major best practices here",
        ]
        for b in backfill:
            if b not in chips:
                chips.append(b)
            if len(chips) >= 3:
                break
    
    chips = chips[:5]  # cap to keep UI tidy

    # If refusal and we have good context, try quick salvage (but limit this to save time)
    if is_refusal(ans) and strong and len(context) > 500 and not farewell:
        try:
            salvage = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role":"system","content":"Write PLAIN TEXT only. No markdown. Be concise."},
                    {"role":"user","content": f"Summarize in 3 short hyphen bullets:\n\n{context[:2000]}"}  # Limit context
                ],
                temperature=0,
                max_tokens=300  # Short salvage response
            )
            ans = strip_md(salvage.choices[0].message.content.strip())
        except Exception:
            pass  # If salvage fails, use original answer

    return ans, chips

# Backward compatibility - loads index each time (slower)
def answer(query: str, top_k=6, prev_chips: Optional[List[str]] = None, last_answer: str = ""):
    """Legacy function that loads index from disk - use answer_with_index for better performance"""
    embs, meta = load_index()
    return answer_with_index(embs, meta, query, top_k, prev_chips, last_answer)

# ------------- index version for UI resets -------------
def index_version() -> str:
    try:
        meta_path = os.path.join(INDEX_DIR, "meta.json")
        emb_path  = os.path.join(INDEX_DIR, "embeddings.npy")
        with open(meta_path, "rb") as f:
            meta_bytes = f.read()
        embs = np.load(emb_path, mmap_mode="r")  # Memory-map for speed
        shape = json.dumps({"shape": embs.shape}, sort_keys=True).encode()
        h = hashlib.sha1(); h.update(meta_bytes); h.update(shape)
        return h.hexdigest()
    except Exception:
        return f"fallback-{int(time.time())}"