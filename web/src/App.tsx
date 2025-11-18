import type { FormEvent } from "react";
import { useState } from "react";

type Mode = "chunks" | "generate";

interface Chunk {
	title: string;
	url: string;
	snippet: string;
}

interface Citation {
	title: string;
	url: string;
}

interface QueryChunksResponse {
	chunks: Chunk[];
}

interface QueryGenerateResponse {
	answer: string;
	citations: Citation[];
}

const initialPlaceholder =
	"Ask about coastal flooding, resilience, cyber security or net zero strategy…";

function App() {
	const [q, setQ] = useState("");
	const [mode, setMode] = useState<Mode>("generate");
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const [chunks, setChunks] = useState<Chunk[]>([]);
	const [answer, setAnswer] = useState("");
	const [citations, setCitations] = useState<Citation[]>([]);

	const hasAnswer = mode === "generate" && answer.length > 0;
	const hasChunks = chunks.length > 0;
	const hasAnyResult = hasAnswer || hasChunks;

	async function fetchChunks(question: string): Promise<void> {
		const resp = await fetch("/api/query", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ q: question, mode: "chunks" }),
		});

		if (!resp.ok) {
			const detail =
				(await resp.json().catch(() => null)) as { detail?: string } | null;
			throw new Error(detail?.detail || `Request failed (${resp.status})`);
		}

		const data = (await resp.json()) as QueryChunksResponse;
		setChunks(data.chunks ?? []);
	}

	async function fetchAnswer(question: string): Promise<void> {
		const resp = await fetch("/api/query", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ q: question, mode: "generate" }),
		});

		if (!resp.ok) {
			const detail =
				(await resp.json().catch(() => null)) as { detail?: string } | null;
			throw new Error(detail?.detail || `Request failed (${resp.status})`);
		}

		const data = (await resp.json()) as QueryGenerateResponse;
		setAnswer(data.answer ?? "");
		setCitations(data.citations ?? []);
	}

	async function handleSubmit(e: FormEvent) {
		e.preventDefault();
		const trimmed = q.trim();
		if (!trimmed || loading) {
			return;
		}

		setLoading(true);
		setError(null);
		setChunks([]);
		setAnswer("");
		setCitations([]);

		try {
			if (mode === "chunks") {
				await fetchChunks(trimmed);
			} else {
				// Generative mode: show answer *and* retrieved chunks
				await Promise.all([fetchChunks(trimmed), fetchAnswer(trimmed)]);
			}
		} catch (err) {
			const msg =
				err instanceof Error
					? err.message
					: "Something went wrong talking to the API.";
			setError(msg);
		} finally {
			setLoading(false);
		}
	}

	return (
		<main className="app-root">
			<header>
				<h1>
					GOV.UK RAG <span>(Lite)</span>
				</h1>
				<p className="subtitle">Ask a question about the seeded GOV.UK docs.</p>
			</header>

			<section className="chat-shell">
				<div className="chat-messages">
					{error && (
						<div className="chat-bubble chat-bubble-error">
							<strong>Error</strong>
							<p>{error}</p>
						</div>
					)}

					{hasAnswer && (
						<div className="chat-bubble chat-bubble-answer">
							<strong>Answer</strong>
							<p>{answer}</p>

							{citations.length > 0 && (
								<div className="chat-citations">
									<p className="chat-citations-label">Sources:</p>
									<ul>
										{citations.map((c, idx) => (
											<li key={`${idx}-${c.title}-${c.url}`}>
												{c.title || "Unknown source"}
												{c.url && (
													<>
														{" "}
														(<a
															href={c.url}
															target="_blank"
															rel="noreferrer"
														>
															open
														</a>
														)
													</>
												)}
											</li>
										))}
									</ul>
								</div>
							)}
						</div>
					)}

					{hasChunks && (
						<div className="chat-bubble chat-bubble-chunks">
							<strong>Retrieved chunks</strong>
							<ul className="chunk-list">
								{chunks.map((c, idx) => (
									<li key={`${idx}-${c.title}-${c.url}`} className="chunk-item">
										<div className="chunk-meta">
											<span className="chunk-title">
												{c.title || `Chunk ${idx + 1}`}
											</span>
											{c.url && (
												<a href={c.url} target="_blank" rel="noreferrer">
													Open document
												</a>
											)}
										</div>
										<p className="chunk-snippet">{c.snippet}</p>
									</li>
								))}
							</ul>
						</div>
					)}

					{!hasAnyResult && !error && null}
				</div>

				<form className="chat-form" onSubmit={handleSubmit}>
					<div className="chat-form-left">
						<div className="mode-switch">
							<label>
								<input
									type="radio"
									name="mode"
									value="generate"
									checked={mode === "generate"}
									onChange={() => setMode("generate")}
								/>
								<span>Generative</span>
							</label>
							<label>
								<input
									type="radio"
									name="mode"
									value="chunks"
									checked={mode === "chunks"}
									onChange={() => setMode("chunks")}
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
								placeholder={initialPlaceholder}
								value={q}
								onChange={(e) => setQ(e.target.value)}
								autoComplete="off"
							/>
						</div>
					</div>

					<button type="submit" disabled={loading || !q.trim()}>
						{loading ? "Asking…" : "Ask"}
					</button>
				</form>
			</section>
		</main>
	);
}

export default App;
