import type { CorpusStatus, Mode, QueryResponse, RetrievalMode } from "../types";

export class ApiError extends Error {}

export async function fetchCorpusStatus(): Promise<CorpusStatus> {
  const resp = await fetch("/api/status").catch(() => {
    throw new ApiError("Unable to reach the API.");
  });
  if (!resp.ok) {
    throw new ApiError(`Status request failed (${resp.status})`);
  }
  try {
    return (await resp.json()) as CorpusStatus;
  } catch {
    throw new ApiError("Invalid status response from the API.");
  }
}

export function localAsOfToIso(value: string): string | undefined {
  if (!value) return undefined;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) throw new ApiError("Choose a valid observed-as-of time.");
  return date.toISOString();
}

export async function queryApi(
  question: string,
  selectedMode: Mode,
  retrievalMode: RetrievalMode,
  asOf?: string,
): Promise<QueryResponse> {
  const resp = await fetch("/api/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      q: question,
      mode: selectedMode,
      retrieval_mode: retrievalMode,
      ...(asOf ? { as_of: asOf } : {}),
    }),
  }).catch(() => {
    throw new ApiError("Unable to reach the API.");
  });

  if (!resp.ok) {
    const detail = (await resp.json().catch(() => null)) as { detail?: unknown } | null;
    throw new ApiError(typeof detail?.detail === "string" ? detail.detail : `Request failed (${resp.status})`);
  }

  let data: QueryResponse;
  try {
    data = (await resp.json()) as QueryResponse;
  } catch {
    throw new ApiError("Invalid response from the API.");
  }
  return {
    ...data,
    answer: data.answer ?? "",
    citations: data.citations ?? [],
    claims: data.claims ?? [],
    chunks: data.chunks ?? [],
  };
}
