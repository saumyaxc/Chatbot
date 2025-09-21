import { useEffect, useLayoutEffect, useRef, useState } from "react";
import "./chat.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const DEFAULT_CHIPS = []; // server supplies chips

function TypingDots() {
  return (
    <span className="zcb-typing" aria-label="Assistant is typing">
      <span className="dot" />
      <span className="dot" />
      <span className="dot" />
    </span>
  );
}

function Message({ role, text }) {
  const cls = `zcb-msg ${role === "user" ? "user" : "bot"}`;
  return (
    <div className={cls}>
      <div>{text}</div>
    </div>
  );
}

export default function ChatBubble() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hi! How can I help you today?" },
  ]);
  const [chips, setChips] = useState(DEFAULT_CHIPS);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const panelRef = useRef(null);
  const messagesRef = useRef(null);
  const inputWrapRef = useRef(null);

  const [atBottom, setAtBottom] = useState(true);
  const [autoFillActive, setAutoFillActive] = useState(false);
  const [inputHeight, setInputHeight] = useState(96);

  // Track all chips shown this session to avoid repeats (case-insensitive)
  const chipHistoryRef = useRef(new Set());
  const indexVersionRef = useRef(null);

  function measureInputHeight() {
    const el = inputWrapRef.current;
    if (!el) return;
    setInputHeight(el.offsetHeight + 12);
  }
  useLayoutEffect(() => { measureInputHeight(); }, []);
  useEffect(() => {
    const ro = new ResizeObserver(measureInputHeight);
    if (inputWrapRef.current) ro.observe(inputWrapRef.current);
    const onResize = () => measureInputHeight();
    window.addEventListener("resize", onResize);
    return () => { ro.disconnect(); window.removeEventListener("resize", onResize); };
  }, []);

  function computeAtBottom() {
    const el = messagesRef.current;
    if (!el) return true;
    const nearBottom = el.scrollHeight - (el.scrollTop + el.clientHeight) <= 8;
    setAtBottom(nearBottom);
    return nearBottom;
  }
  function scrollToBottom({ smooth = true } = {}) {
    const el = messagesRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }
  function isOverflowing() {
    const el = messagesRef.current;
    if (!el) return false;
    return el.scrollHeight > el.clientHeight + 1;
  }

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    function onClick(e) {
      if (!open) return;
      if (panelRef.current && !panelRef.current.contains(e.target)) {
        if (e.target.closest?.("#zigtal-bubble-btn")) return;
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => {
      scrollToBottom({ smooth: false });
      computeAtBottom();
    }, 0);
    return () => clearTimeout(t);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const el = messagesRef.current;
    if (!el) return;
    const onScroll = () => {
      const nearBottom = computeAtBottom();
      if (!nearBottom) setAutoFillActive(false);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    if (autoFillActive || atBottom) {
      scrollToBottom({ smooth: !autoFillActive });
      if (autoFillActive && isOverflowing()) setAutoFillActive(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length]);

  useEffect(() => {
    if (!open) return;
    if (loading && (atBottom || autoFillActive)) {
      const t = setTimeout(() => scrollToBottom({ smooth: true }), 0);
      return () => clearTimeout(t);
    }
  }, [loading, open, atBottom, autoFillActive]);

  async function fetchIndexVersion() {
    try {
      const res = await fetch(`${API_BASE}/index_version`);
      const data = await res.json();
      return data?.version || null;
    } catch {
      return null;
    }
  }

  useEffect(() => {
    if (!open) return;
    (async () => {
      const v = await fetchIndexVersion();
      if (v && v !== indexVersionRef.current) {
        indexVersionRef.current = v;
        chipHistoryRef.current = new Set(); // clear chip memory
        setChips([]);
        setMessages([{ role: "assistant", text: "Hi! How can I help you today?" }]);
      }
    })();
  }, [open]);

  function normalizedChipSet() {
    return Array.from(chipHistoryRef.current);
  }

  function addChipsToHistory(candidates = []) {
    for (const c of candidates) {
      const key = (c || "").trim().toLowerCase();
      if (key) chipHistoryRef.current.add(key);
    }
  }

  async function send(textOverride) {
    const q = (textOverride ?? input).trim();
    if (!q || loading) return;

    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setLoading(true);

    const lastAssistant = [...messages].reverse().find(m => m.role === "assistant")?.text || "";

    if (!isOverflowing()) setAutoFillActive(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          prev_chips: normalizedChipSet(),
          last_answer: lastAssistant,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");

      if (data.version && data.version !== indexVersionRef.current) {
        indexVersionRef.current = data.version;
        chipHistoryRef.current = new Set();
      }

      const assistantMsg = { role: "assistant", text: data.answer };
      setMessages((m) => [...m, assistantMsg]);

      // server-provided chips with client-side dedupe
      const incoming = Array.isArray(data.chips) ? data.chips : [];
      const next = [];
      const seen = new Set();
      for (const c of incoming) {
        const clean = (c || "").trim();
        const key = clean.toLowerCase();
        if (!clean) continue;
        if (seen.has(key)) continue;
        if (chipHistoryRef.current.has(key)) continue;
        seen.add(key);
        next.push(clean);
      }
      setChips(next);
      addChipsToHistory(next);
    } catch (e) {
      const errMsg = { role: "assistant", text: `Error: ${e.message}` };
      setMessages((m) => [...m, errMsg]);
      setChips([]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  const last = messages[messages.length - 1];
  const showInlineChips = !loading && last?.role === "assistant" && chips.length > 0;

  return (
    <>
      {/* Floating bubble (logo in /public) */}
      <button
        id="zigtal-bubble-btn"
        className="zcb-bubble"
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "Close chat" : "Open chat"}
        title="Chat with Zigtal"
      >
        <img src="/zigtal-logo.png" alt="Zigtal" />
      </button>

      {/* Panel */}
      <div ref={panelRef} className={`zcb-panel ${open ? "open" : ""}`} role="dialog" aria-modal="false">
        <div className="zcb-header">
          <div className="zcb-headerTitle">
            <span className="zcb-dotBrand" />
            <span>Zigtal Chat</span>
          </div>
          <div></div>
        </div>

        {/* Messages + typing bubble + inline chips */}
        <div className="zcb-messages" ref={messagesRef}>
          {messages.map((m, i) => (
            <Message key={i} role={m.role} text={m.text} />
          ))}

          {loading && (
            <div className="zcb-typingBubble">
              <TypingDots />
            </div>
          )}

          {showInlineChips && (
            <div className="zcb-chipsInline" aria-label="Quick replies">
              {chips.map((q, idx) => (
                <button key={`${q}-${idx}`} className="zcb-chip" onClick={() => send(q)} disabled={loading}>
                  {q}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Down button — centered above input */}
        {open && !atBottom && (
          <button
            className="zcb-toBottom"
            style={{ bottom: inputHeight }}
            onClick={() => scrollToBottom({ smooth: true })}
            aria-label="Scroll to bottom"
            title="Scroll to bottom"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>
        )}

        {/* Input */}
        <div className="zcb-inputWrap" ref={inputWrapRef}>
          <textarea
            className="zcb-textarea"
            rows={2}
            placeholder="Type your question…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <button
            className="zcb-sendBtn"
            onClick={() => send()}
            disabled={loading || !input.trim()}
          >
            Send
          </button>
        </div>
      </div>
    </>
  );
}