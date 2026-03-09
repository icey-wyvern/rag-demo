import type { FormEvent } from "react";

import type { Mode } from "../types";

interface ChatComposerProps {
  mode: Mode;
  loading: boolean;
  question: string;
  corpusReady: boolean;
  onModeChange: (mode: Mode) => void;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
}

export function ChatComposer({
  mode,
  loading,
  question,
  corpusReady,
  onModeChange,
  onQuestionChange,
  onSubmit,
}: ChatComposerProps) {
  const disabled = loading || !question.trim() || !corpusReady;

  return (
    <form className="chat-form" onSubmit={onSubmit}>
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
