// src/App.jsx
import ChatBubble from "./ChatBubble.jsx";
export default function App() {
  return (
    <div style={{ minHeight: "100vh", background: "#ffffff" }}>
      <div style={{ padding: 20 }}>
        <h1 style={{ margin: 0, color: "#0f172a" }}>Prototype</h1>
        <p style={{ color: "#334155" }}>Click the chat bubble in the bottom-right.</p>
      </div>
      <ChatBubble />
    </div>
  );
}
