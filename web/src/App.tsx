import type { FormEvent } from "react";
import { useEffect, useRef, useState } from "react";

import { ChatComposer } from "./components/ChatComposer";
import { ChatMessages } from "./components/ChatMessages";
import { StatusBanner } from "./components/StatusBanner";
import { ApiError, fetchCorpusStatus, localAsOfToIso, queryApi } from "./lib/api";
import type { ChatMessage, CorpusStatus, Mode, RetrievalMode } from "./types";

const initialMessages: ChatMessage[] = [{ id: 1, role: "assistant", kind: "system" }];

function nextMessageId(offset = 0): number {
  return Date.now() + offset;
}

function App() {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<Mode>("generate");
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("hybrid");
  const [asOfLocal, setAsOfLocal] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [corpusStatus, setCorpusStatus] = useState<CorpusStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const corpusReady = retrievalMode === "lexical"
    ? Boolean(corpusStatus?.versioned?.lexical_ready)
    : Boolean(corpusStatus?.is_ready);
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const lastUserMessageRef = useRef<HTMLDivElement | null>(null);
  const submittingRef = useRef(false);

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
          setStatusError(error instanceof ApiError ? error.message : "Unable to reach the API.");
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
    if (!trimmed || loading || submittingRef.current || !corpusReady) {
      return;
    }
    submittingRef.current = true;

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
      const result = await queryApi(trimmed, mode, retrievalMode, localAsOfToIso(asOfLocal));
      const nextMessages: ChatMessage[] = [];

      if (mode === "generate" && (result.status === "answered" || result.status === "abstained")) {
        nextMessages.push({
          id: nextMessageId(1),
          role: "assistant",
          kind: "answer",
          content: result.answer,
          result,
        });
      }

      if (mode === "chunks") {
        nextMessages.push({
          id: nextMessageId(2),
          role: "assistant",
          kind: "chunks",
          result,
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
          content: error instanceof ApiError ? error.message : "Something went wrong talking to the API.",
        },
      ]);
    } finally {
      setLoading(false);
      submittingRef.current = false;
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
        <p className="subtitle">Ask about locally observed GOV.UK guidance, with evidence for each answer.</p>
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
          retrievalMode={retrievalMode}
          asOfLocal={asOfLocal}
          loading={loading}
          question={q}
          corpusReady={corpusReady}
          onModeChange={setMode}
          onRetrievalModeChange={setRetrievalMode}
          onAsOfLocalChange={setAsOfLocal}
          onQuestionChange={setQ}
          onSubmit={handleSubmit}
        />
      </section>
    </main>
  );
}

export default App;
