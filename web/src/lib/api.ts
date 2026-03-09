import type { CorpusStatus, Mode, QueryResponse } from "../types";

export async function fetchCorpusStatus(): Promise<CorpusStatus> {
  const resp = await fetch("/api/status");
  if (!resp.ok) {
    throw new Error(`Status request failed (${resp.status})`);
  }
  return (await resp.json()) as CorpusStatus;
}

export async function queryApi(question: string, selectedMode: Mode): Promise<QueryResponse> {
  const resp = await fetch("/api/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ q: question, mode: selectedMode }),
  });

  if (!resp.ok) {
    const detail = (await resp.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(detail?.detail || `Request failed (${resp.status})`);
  }

  const data = (await resp.json()) as QueryResponse;
  return {
    answer: data.answer ?? "",
    citations: data.citations ?? [],
    chunks: data.chunks ?? [],
  };
}
