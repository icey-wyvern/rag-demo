import type { FormEvent } from "react";
import { useEffect, useRef, useState } from "react";

import { ChatComposer } from "./components/ChatComposer";
import { ChatMessages } from "./components/ChatMessages";
import { StatusBanner } from "./components/StatusBanner";
import { fetchCorpusStatus, queryApi } from "./lib/api";
import type { ChatMessage, CorpusStatus, Mode } from "./types";

const initialMessages: ChatMessage[] = [{ id: 1, role: "assistant", kind: "system" }];

function nextMessageId(offset = 0): number {
  return Date.now() + offset;
}

function App() {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<Mode>("generate");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [corpusStatus, setCorpusStatus] = useState<CorpusStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const lastUserMessageRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      try {
        const status = await fetchCorpusStatus();
        if (!cancelled) {
          setCorpusStatus(status);
          setStatusError("");
        }
      } catch (error) {
        if (!cancelled) {
          setStatusError(error instanceof Error ? error.message : "Unable to reach the API.");
        }
      }
    }

    void loadStatus();
    const intervalId = window.setInterval(() => {
      void loadStatus();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = q.trim();
    if (!trimmed || loading || !corpusStatus?.is_ready) {
      return;
    }

    setMessages((prev) => [
      ...prev,
      {
        id: nextMessageId(),
        role: "user",
        content: trimmed,
      },
    ]);
    setQ("");
    setLoading(true);

    try {
      const result = await queryApi(trimmed, mode);
      const nextMessages: ChatMessage[] = [];

      if (mode === "generate" && result.answer) {
        nextMessages.push({
          id: nextMessageId(1),
          role: "assistant",
          kind: "answer",
          content: result.answer,
          citations: result.citations,
        });
      }

      if (result.chunks.length > 0) {
        nextMessages.push({
          id: nextMessageId(2),
          role: "assistant",
          kind: "chunks",
          chunks: result.chunks,
        });
      }

      if (!nextMessages.length) {
        nextMessages.push({
          id: nextMessageId(3),
          role: "error",
          content: "No matching content was returned.",
        });
      }

      setMessages((prev) => [...prev, ...nextMessages]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId(),
          role: "error",
          content: error instanceof Error ? error.message : "Something went wrong talking to the API.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!messages.length) {
      return;
    }

    const last = messages[messages.length - 1];
    if (last.role !== "assistant") {
      return;
    }

    const container = messagesContainerRef.current;
    const lastUser = lastUserMessageRef.current;
    if (!container || !lastUser) {
      return;
    }

    const margin = container.clientHeight * 0.1;
    container.scrollTo({
      top: Math.max(lastUser.offsetTop - margin, 0),
      behavior: "smooth",
    });
  }, [messages]);

  return (
    <main className="app-root">
      <header>
        <h1>GOV.UK RAG</h1>
        <p className="subtitle">Ask grounded questions about the seeded GOV.UK policy and resilience documents.</p>
      </header>

      <StatusBanner status={corpusStatus} statusError={statusError} />

      <section className="chat-shell">
        <ChatMessages
          messages={messages}
          messagesContainerRef={messagesContainerRef}
          lastUserMessageRef={lastUserMessageRef}
        />

        <ChatComposer
          mode={mode}
          loading={loading}
          question={q}
          corpusReady={Boolean(corpusStatus?.is_ready)}
          onModeChange={setMode}
          onQuestionChange={setQ}
          onSubmit={handleSubmit}
        />
      </section>
    </main>
  );
}

export default App;
