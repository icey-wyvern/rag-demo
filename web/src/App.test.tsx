import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { QueryResponse } from "./types";

const readyStatus = {
  status: "ready", ingest_version: 1, documents_total: 1, documents_ready: 1,
  last_error: null, is_ready: true,
  versioned: { schema_ready: true, sources: 1, snapshots: 2, current_passages: 2,
    embedded_passages: 2, lexical_ready: true, semantic_ready: true, provider_configured: true },
};

const citation = {
  evidence_id: "E1", supporting_quote: "Exact supporting words.", title: "Service guidance",
  url: "https://www.gov.uk/guidance/service", source_key: "service",
  snapshot_id: "7", snapshot_content_sha256: "snapshot-hash", observed_at: "2026-10-15T09:00:00Z",
  source_updated_at: "2026-10-14T12:00:00Z", passage_id: "12", passage_ordinal: 0,
  passage_sha256: "passage-hash", page: null,
};

const answered: QueryResponse = {
  status: "answered", answer: "The service requires a review.",
  citations: [citation, citation], sources: [citation],
  claims: [{ text: "A review is required.", citations: [citation, citation] }],
  chunks: [{ ...citation, snippet: "Exact supporting words.", text: "Exact supporting words." }],
  retrieval: { retrieval_mode: "hybrid", k: 5, as_of: null, evidence_count: 1 },
};

function mockApi(result: QueryResponse = answered, status = readyStatus) {
  const requests: Array<{ url: string; body?: Record<string, unknown> }> = [];
  vi.spyOn(window, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    requests.push({ url, body: init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined });
    return new Response(JSON.stringify(url === "/api/status" ? status : result), { status: 200 });
  });
  return requests;
}

async function submitQuestion() {
  await screen.findByText(/Versioned corpus ready/i);
  fireEvent.change(screen.getByLabelText("Question"), { target: { value: "What changed?" } });
  fireEvent.click(screen.getByRole("button", { name: "Ask" }));
  await waitFor(() => expect(screen.getByText("What changed?")).toBeInTheDocument());
}

describe("App", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    Element.prototype.scrollTo = vi.fn();
  });
  afterEach(() => cleanup());

  it("keeps the readiness gate when the corpus is not ready", async () => {
    mockApi(answered, { ...readyStatus, status: "running", is_ready: false });
    render(<App />);
    expect(await screen.findByText(/Versioned corpus setup in progress/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
  });

  it("allows lexical inspection while semantic setup is still in progress", async () => {
    mockApi(answered, { ...readyStatus, status: "running", is_ready: false });
    render(<App />);
    await screen.findByText(/Versioned corpus setup in progress/i);
    fireEvent.change(screen.getByLabelText("Retrieval mode"), { target: { value: "lexical" } });
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "What changed?" } });
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
  });

  it("defaults to hybrid and omits a blank as-of", async () => {
    const requests = mockApi();
    render(<App />);
    await submitQuestion();
    await screen.findByText("The service requires a review.");
    expect(requests.find((request) => request.url === "/api/query")?.body).toEqual({
      q: "What changed?", mode: "generate", retrieval_mode: "hybrid",
    });
  });

  it.each(["lexical", "semantic"] as const)("sends selected %s retrieval", async (mode) => {
    const requests = mockApi();
    render(<App />);
    fireEvent.change(screen.getByLabelText("Retrieval mode"), { target: { value: mode } });
    await submitQuestion();
    await waitFor(() => expect(requests.find((request) => request.url === "/api/query")?.body?.retrieval_mode).toBe(mode));
  });

  it("converts local observed-as-of time into a timezone-aware ISO timestamp and can clear it", async () => {
    const requests = mockApi();
    render(<App />);
    fireEvent.change(screen.getByLabelText("Observed as of"), { target: { value: "2026-10-15T10:30" } });
    expect(screen.getByRole("button", { name: "Clear" })).toBeInTheDocument();
    await submitQuestion();
    const sent = requests.find((request) => request.url === "/api/query")?.body?.as_of;
    expect(sent).toBe(new Date("2026-10-15T10:30").toISOString());
    expect(sent).toMatch(/(?:Z|[+-]\d\d:\d\d)$/);
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByLabelText("Observed as of")).toHaveValue("");
  });

  it("shows answered claims, deduplicated exact evidence, and provenance", async () => {
    mockApi();
    render(<App />);
    await submitQuestion();
    expect(await screen.findByText("The service requires a review.")).toBeInTheDocument();
    expect(screen.getByText("Answered")).toBeInTheDocument();
    expect(screen.getByText("A review is required.")).toBeInTheDocument();
    expect(screen.getAllByText("Exact supporting words.")).toHaveLength(1);
    expect(screen.getAllByRole("link", { name: "Service guidance" })).toHaveLength(1);
    expect(screen.getByText(/Observed:/)).toBeInTheDocument();
    expect(screen.getByText(/Source updated:/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Technical provenance"));
    expect(screen.getByText("snapshot-hash")).toBeInTheDocument();
    expect(screen.queryByText("Retrieved passages")).not.toBeInTheDocument();
  });

  it("shows as-of context from retrieval metadata", async () => {
    mockApi({ ...answered, retrieval: { ...answered.retrieval!, as_of: "2026-10-15T09:00:00Z" } });
    render(<App />);
    await submitQuestion();
    expect(await screen.findByText(/Hybrid retrieval.*Observed as of/)).toBeInTheDocument();
  });

  it("renders abstention as a successful result without supporting evidence", async () => {
    mockApi({ ...answered, status: "abstained", answer: "Insufficient evidence.",
      abstention_reason: "Insufficient evidence.", claims: [], citations: [], sources: [] });
    render(<App />);
    await submitQuestion();
    expect(await screen.findByText("Abstained")).toBeInTheDocument();
    expect(screen.getByText("Insufficient evidence.")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Service guidance" })).not.toBeInTheDocument();
    expect(screen.queryByText("Request failed")).not.toBeInTheDocument();
  });

  it("preserves chunks mode as retrieval inspection", async () => {
    const requests = mockApi({ ...answered, status: "retrieved", answer: "", claims: [], citations: [] });
    render(<App />);
    fireEvent.click(screen.getByLabelText("Chunks only"));
    await submitQuestion();
    expect(await screen.findByText("Retrieved passages")).toBeInTheDocument();
    expect(screen.getByText(/not validated answer citations/i)).toBeInTheDocument();
    expect(within(screen.getByText("Retrieved passages").parentElement!).getByText("Exact supporting words.")).toBeInTheDocument();
    expect(screen.queryByText("Validated claims and evidence")).not.toBeInTheDocument();
    expect(requests.find((request) => request.url === "/api/query")?.body?.mode).toBe("chunks");
  });

  it("shows sanitized API errors without an answered result", async () => {
    vi.spyOn(window, "fetch").mockImplementation(async (input) => new Response(
      JSON.stringify(String(input) === "/api/status" ? readyStatus : { detail: "Query service unavailable." }),
      { status: String(input) === "/api/status" ? 200 : 503 },
    ));
    render(<App />);
    await submitQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent("Query service unavailable.");
    expect(screen.queryByText("Answered")).not.toBeInTheDocument();
  });

  it("does not display arbitrary error objects", async () => {
    vi.spyOn(window, "fetch").mockImplementation(async (input) => new Response(
      JSON.stringify(String(input) === "/api/status" ? readyStatus : { detail: { internal: "hidden" } }),
      { status: String(input) === "/api/status" ? 200 : 422 },
    ));
    render(<App />);
    await submitQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent("Request failed (422)");
    expect(screen.queryByText("hidden")).not.toBeInTheDocument();
  });
});
