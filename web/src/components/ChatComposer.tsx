import type { FormEvent } from "react";

import type { Mode, RetrievalMode } from "../types";

interface ChatComposerProps {
  mode: Mode;
  retrievalMode: RetrievalMode;
  asOfLocal: string;
  loading: boolean;
  question: string;
  corpusReady: boolean;
  onModeChange: (mode: Mode) => void;
  onRetrievalModeChange: (mode: RetrievalMode) => void;
  onAsOfLocalChange: (value: string) => void;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
}

export function ChatComposer({
  mode,
  retrievalMode,
  asOfLocal,
  loading,
  question,
  corpusReady,
  onModeChange,
  onRetrievalModeChange,
  onAsOfLocalChange,
  onQuestionChange,
  onSubmit,
}: ChatComposerProps) {
  const disabled = loading || !question.trim() || !corpusReady;

  return (
    <form className="chat-form" onSubmit={onSubmit}>
      <div className="query-options">
        <div className="retrieval-option">
          <label htmlFor="retrieval-mode">Retrieval mode</label>
          <select
            id="retrieval-mode"
            value={retrievalMode}
            onChange={(event) => onRetrievalModeChange(event.target.value as RetrievalMode)}
            disabled={loading}
          >
            <option value="hybrid">Hybrid</option>
            <option value="semantic">Semantic</option>
            <option value="lexical">Lexical</option>
          </select>
        </div>
        <div className="as-of-option">
          <label htmlFor="as-of">Observed as of</label>
          <div className="as-of-controls">
            <input
              id="as-of"
              type="datetime-local"
              value={asOfLocal}
              onChange={(event) => onAsOfLocalChange(event.target.value)}
              aria-describedby="as-of-help"
              disabled={loading}
            />
            {asOfLocal && <button type="button" className="clear-button" onClick={() => onAsOfLocalChange("")}>Clear</button>}
          </div>
          <small id="as-of-help">Optional: query the latest version observed by this time.</small>
        </div>
      </div>
      <div className="chat-form-left">
        <div className="mode-switch">
          <label>
            <input
              type="radio"
              name="mode"
              value="generate"
              checked={mode === "generate"}
              onChange={() => onModeChange("generate")}
            />
            <span>Generative</span>
          </label>
          <label>
            <input
              type="radio"
              name="mode"
              value="chunks"
              checked={mode === "chunks"}
              onChange={() => onModeChange("chunks")}
            />
            <span>Chunks only</span>
          </label>
        </div>

        <div className="chat-input-wrapper">
          <label className="sr-only" htmlFor="question">
            Question
          </label>
          <input
            id="question"
            type="text"
            placeholder={corpusReady ? "Ask a question..." : "Wait for corpus setup to finish..."}
            value={question}
            onChange={(event) => onQuestionChange(event.target.value)}
            autoComplete="off"
            disabled={!corpusReady}
          />
        </div>
      </div>

      <button type="submit" disabled={disabled}>
        {loading ? "Asking..." : "Ask"}
      </button>
    </form>
  );
}
