import type { RefObject } from "react";

import type { ChatMessage, Citation, Chunk } from "../types";

function buildSourceUrl(url: string, page?: number): string {
  if (!url) {
    return url;
  }
  return page ? `${url}#page=${page}` : url;
}

function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations.length) {
    return null;
  }

  return (
    <div className="chat-citations">
      <p className="chat-citations-label">Sources:</p>
      <ul>
        {citations.map((citation, index) => (
          <li key={`${index}-${citation.title}-${citation.url}`}>
            {citation.title || "Unknown source"}
            {citation.url && (
              <>
                {" "}
                (
                <a href={buildSourceUrl(citation.url, citation.page)} target="_blank" rel="noreferrer">
                  open
                </a>
                )
              </>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ChunkList({ chunks }: { chunks: Chunk[] }) {
  return (
    <ul className="chunk-list">
      {chunks.map((chunk, index) => (
        <li key={`${index}-${chunk.title}-${chunk.url}`} className="chunk-item">
          <div className="chunk-meta">
            <span className="chunk-title">{chunk.title || `Chunk ${index + 1}`}</span>
            {chunk.url && (
              <a href={buildSourceUrl(chunk.url, chunk.page)} target="_blank" rel="noreferrer">
                Open document
              </a>
            )}
          </div>
          <p className="chunk-snippet">{chunk.snippet}</p>
        </li>
      ))}
    </ul>
  );
}

const SEEDED_DOCS = [
  "National Risk Register 2025",
  "The UK Government Resilience Framework",
  "British Energy Security Strategy",
  "Government Cyber Security Strategy 2022-2030",
  "Net Zero Strategy: Build Back Greener",
];

const EXAMPLE_QUESTIONS = [
  "How does the National Risk Register describe the main impacts of coastal flooding?",
  "What are the key principles of resilience in the UK Government Resilience Framework?",
  "What are the main goals of the Net Zero Strategy: Build Back Greener?",
];

export function ChatMessages({
  messages,
  messagesContainerRef,
  lastUserMessageRef,
}: {
  messages: ChatMessage[];
  messagesContainerRef: RefObject<HTMLDivElement | null>;
  lastUserMessageRef: RefObject<HTMLDivElement | null>;
}) {
  return (
    <div className="chat-messages" ref={messagesContainerRef}>
      {messages.map((msg) => {
        if (msg.role === "assistant" && msg.kind === "system") {
          return (
            <div key={msg.id} className="chat-message-row chat-message-row-assistant">
              <div className="chat-bubble">
                <strong>Assistant</strong>
                <p>You can ask questions about the following seeded GOV.UK documents:</p>
                <ul className="seeded-list">
                  {SEEDED_DOCS.map((doc) => (
                    <li key={doc}>{doc}</li>
                  ))}
                </ul>
                <p className="seeded-examples-label">Example questions:</p>
                <ul className="seeded-list">
                  {EXAMPLE_QUESTIONS.map((question) => (
                    <li key={question}>{question}</li>
                  ))}
                </ul>
              </div>
            </div>
          );
        }

        const bubbleClassName = [
          "chat-bubble",
          msg.role === "assistant" && msg.kind === "chunks" ? "chat-bubble-chunks" : "",
          msg.role === "assistant" && msg.kind === "answer" ? "chat-bubble-answer" : "",
          msg.role === "error" ? "chat-bubble-error" : "",
        ]
          .filter(Boolean)
          .join(" ");

        return (
          <div
            key={msg.id}
            className={`chat-message-row chat-message-row-${msg.role}`}
            ref={(element) => {
              if (msg.role === "user" && element) {
                lastUserMessageRef.current = element;
              }
            }}
          >
            {msg.role === "user" && (
              <div className={bubbleClassName}>
                <strong>You</strong>
                <p>{msg.content}</p>
              </div>
            )}

            {msg.role === "assistant" && msg.kind === "answer" && (
              <div className={bubbleClassName}>
                <strong>Answer</strong>
                <p>{msg.content}</p>
                <CitationList citations={msg.citations ?? []} />
              </div>
            )}

            {msg.role === "assistant" && msg.kind === "chunks" && msg.chunks && (
              <div className={bubbleClassName}>
                <strong>Retrieved chunks</strong>
                <ChunkList chunks={msg.chunks} />
              </div>
            )}

            {msg.role === "error" && (
              <div className={bubbleClassName}>
                <strong>Error</strong>
                <p>{msg.content}</p>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
