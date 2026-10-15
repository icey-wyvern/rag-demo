import type { RefObject } from "react";

import type { ChatMessage, Citation, ClaimCitation, Chunk, Provenance, QueryResponse } from "../types";

function sourceUrl(url: string, page?: number | null): string | null {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
    return page ? `${parsed.href}#page=${page}` : parsed.href;
  } catch {
    return null;
  }
}

function uniqueBy<T>(items: T[], key: (item: T) => string): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const identity = key(item);
    if (seen.has(identity)) return false;
    seen.add(identity);
    return true;
  });
}

function citationKey(citation: Citation): string {
  return `${citation.snapshot_id ?? ""}:${citation.passage_id ?? ""}:${citation.url}:${citation.page ?? ""}`;
}

function Timestamp({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null;
  return <span>{label && `${label}: `}<time dateTime={value}>{new Date(value).toLocaleString()}</time></span>;
}

function ProvenanceDetails({ item }: { item: Provenance }) {
  if (!item.snapshot_id && !item.passage_id && !item.snapshot_content_sha256 && !item.passage_sha256) return null;
  return (
    <details className="provenance-details">
      <summary>Technical provenance</summary>
      <dl>
        {item.source_key && <><dt>Source key</dt><dd>{item.source_key}</dd></>}
        {item.snapshot_id && <><dt>Snapshot ID</dt><dd>{item.snapshot_id}</dd></>}
        {item.snapshot_content_sha256 && <><dt>Snapshot hash</dt><dd>{item.snapshot_content_sha256}</dd></>}
        {item.passage_id && <><dt>Passage ID</dt><dd>{item.passage_id}</dd></>}
        {item.passage_ordinal !== undefined && <><dt>Passage ordinal</dt><dd>{item.passage_ordinal}</dd></>}
        {item.passage_sha256 && <><dt>Passage hash</dt><dd>{item.passage_sha256}</dd></>}
      </dl>
    </details>
  );
}

function EvidenceItem({ citation }: { citation: ClaimCitation }) {
  const href = sourceUrl(citation.url, citation.page);
  return (
    <li className="evidence-item">
      <div className="evidence-source">
        {href ? <a href={href} target="_blank" rel="noreferrer">{citation.title || "Source"}</a> : citation.title || "Unknown source"}
      </div>
      <blockquote>{citation.supporting_quote}</blockquote>
      <div className="evidence-dates">
        <Timestamp label="Observed" value={citation.observed_at} />
        <Timestamp label="Source updated" value={citation.source_updated_at} />
      </div>
      <ProvenanceDetails item={citation} />
    </li>
  );
}

function AnswerEvidence({ result }: { result: QueryResponse }) {
  if (result.claims.length) {
    return (
      <section className="answer-evidence" aria-label="Validated claims and evidence">
        <h3>Validated claims and evidence</h3>
        <ol className="claim-list">
          {result.claims.map((claim, index) => (
            <li key={`${index}-${claim.text}`} className="claim-item">
              <p>{claim.text}</p>
              <ul className="evidence-list">
                {uniqueBy(claim.citations, (citation) => `${citationKey(citation)}:${citation.supporting_quote}`).map((citation) => (
                  <EvidenceItem key={`${citationKey(citation)}:${citation.supporting_quote}`} citation={citation} />
                ))}
              </ul>
            </li>
          ))}
        </ol>
      </section>
    );
  }
  // Retain source-link compatibility for older responses with no structured claims.
  const citations = uniqueBy(result.citations, citationKey);
  if (!citations.length) return null;
  return <div className="chat-citations"><strong>Sources</strong><ul>{citations.map((citation) => <li key={citationKey(citation)}>{sourceUrl(citation.url, citation.page) ? <a href={sourceUrl(citation.url, citation.page)!} target="_blank" rel="noreferrer">{citation.title || "Source"}</a> : citation.title || "Unknown source"}</li>)}</ul></div>;
}

function QueryContext({ result }: { result: QueryResponse }) {
  const retrieval = result.retrieval;
  if (!retrieval) return null;
  return <p className="query-context">{retrieval.retrieval_mode[0].toUpperCase() + retrieval.retrieval_mode.slice(1)} retrieval{retrieval.as_of ? <> · Observed as of <Timestamp label="" value={retrieval.as_of} /></> : " · Latest observed"}</p>;
}

function ChunkList({ chunks }: { chunks: Chunk[] }) {
  if (!chunks.length) return <p>No matching passages were retrieved.</p>;
  return (
    <ul className="chunk-list">
      {chunks.map((chunk, index) => (
        <li key={`${chunk.snapshot_id ?? ""}:${chunk.passage_id ?? index}`} className="chunk-item">
          <div className="chunk-meta">
            <span className="chunk-title">{chunk.title || `Passage ${index + 1}`}</span>
            {sourceUrl(chunk.url, chunk.page) && <a href={sourceUrl(chunk.url, chunk.page)!} target="_blank" rel="noreferrer">Open source</a>}
          </div>
          <p className="chunk-snippet">{chunk.snippet}</p>
          <div className="evidence-dates"><Timestamp label="Observed" value={chunk.observed_at} /><Timestamp label="Source updated" value={chunk.source_updated_at} /></div>
          <ProvenanceDetails item={chunk} />
        </li>
      ))}
    </ul>
  );
}

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
    <div className="chat-messages" ref={messagesContainerRef} aria-live="polite">
      {messages.map((msg) => {
        if (msg.role === "assistant" && msg.kind === "system") {
          return <div key={msg.id} className="chat-message-row chat-message-row-assistant"><div className="chat-bubble"><strong>Assistant</strong><p>Ask about guidance in the locally observed corpus. Generated answers show the exact text cited for each claim. Choose Chunks only to inspect retrieved passages.</p></div></div>;
        }

        const bubbleClassName = [
          "chat-bubble",
          msg.kind === "chunks" ? "chat-bubble-chunks" : "",
          msg.kind === "answer" ? "chat-bubble-answer" : "",
          msg.result?.status === "abstained" ? "chat-bubble-abstained" : "",
          msg.role === "error" ? "chat-bubble-error" : "",
        ].filter(Boolean).join(" ");

        return (
          <div key={msg.id} className={`chat-message-row chat-message-row-${msg.role}`} ref={(element) => {
            if (msg.role === "user" && element) lastUserMessageRef.current = element;
          }}>
            {msg.role === "user" && <div className={bubbleClassName}><strong>You</strong><p>{msg.content}</p></div>}
            {msg.role === "assistant" && msg.kind === "answer" && msg.result && (
              <div className={bubbleClassName}>
                <strong>{msg.result.status === "abstained" ? "Abstained" : "Answered"}</strong>
                <p className="answer-text">{msg.result.status === "abstained" ? msg.result.abstention_reason || msg.content : msg.content}</p>
                <QueryContext result={msg.result} />
                {msg.result.status === "answered" && <AnswerEvidence result={msg.result} />}
              </div>
            )}
            {msg.role === "assistant" && msg.kind === "chunks" && msg.result && (
              <div className={bubbleClassName}>
                <strong>Retrieved passages</strong>
                <p className="inspection-note">For inspection only; these are not validated answer citations.</p>
                <QueryContext result={msg.result} />
                <ChunkList chunks={msg.result.chunks} />
              </div>
            )}
            {msg.role === "error" && <div className={bubbleClassName} role="alert"><strong>Request failed</strong><p>{msg.content}</p></div>}
          </div>
        );
      })}
    </div>
  );
}
