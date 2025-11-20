import type { FormEvent } from "react";
import { useState } from "react";

type Mode = "chunks" | "generate";

interface Chunk {
	title: string;
	url: string;
	snippet: string;
	page?: number;
}

interface Citation {
	title: string;
	url: string;
	page?: number;
}

interface QueryChunksResponse {
	chunks: Chunk[];
}

interface QueryGenerateResponse {
	answer: string;
	citations: Citation[];
}

type MessageKind = "system" | "answer" | "chunks";

interface ChatMessage {
	id: number;
	role: "user" | "assistant" | "error";
	kind?: MessageKind;
	content?: string;
	chunks?: Chunk[];
	citations?: Citation[];
}

const initialPlaceholder = "Ask a question…";

function buildSourceUrl(url: string, page?: number): string {
	if (!url) {
		return url;
	}
	return page ? `${url}#page=${page}` : url;
}

const initialMessages: ChatMessage[] = [
	{
		id: 1,
		role: "assistant",
		kind: "system",
	},
];

function App() {
	const [q, setQ] = useState("");
	const [mode, setMode] = useState<Mode>("generate");
	const [loading, setLoading] = useState(false);
	const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);

	async function fetchChunks(question: string): Promise<Chunk[]> {
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
		return data.chunks ?? [];
	}

	async function fetchAnswer(
		question: string,
	): Promise<{ answer: string; citations: Citation[] }> {
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
		return {
			answer: data.answer ?? "",
			citations: data.citations ?? [],
		};
	}

	async function handleSubmit(e: FormEvent) {
		e.preventDefault();
		const trimmed = q.trim();
		if (!trimmed || loading) {
			return;
		}

		// Push the user's message into the chat and clear the input
		const userMessage: ChatMessage = {
			id: Date.now(),
			role: "user",
			kind: undefined,
			content: trimmed,
		};
		setMessages((prev) => [...prev, userMessage]);
		setQ("");

		setLoading(true);

		try {
			if (mode === "chunks") {
				const retrievedChunks = await fetchChunks(trimmed);

				if (retrievedChunks.length > 0) {
					const chunksMessage: ChatMessage = {
						id: Date.now() + 1,
						role: "assistant",
						kind: "chunks",
						chunks: retrievedChunks,
					};
					setMessages((prev) => [...prev, chunksMessage]);
				}
			} else {
				// Generative mode: show answer AND retrieved chunks
				const [retrievedChunks, answerResult] = await Promise.all([
					fetchChunks(trimmed),
					fetchAnswer(trimmed),
				]);

				const assistantMessages: ChatMessage[] = [];

				if (answerResult.answer) {
					assistantMessages.push({
						id: Date.now() + 1,
						role: "assistant",
						kind: "answer",
						content: answerResult.answer,
						citations: answerResult.citations,
					});
				}

				if (retrievedChunks.length > 0) {
					assistantMessages.push({
						id: Date.now() + 2,
						role: "assistant",
						kind: "chunks",
						chunks: retrievedChunks,
					});
				}

				if (assistantMessages.length > 0) {
					setMessages((prev) => [...prev, ...assistantMessages]);
				}
			}
		} catch (err) {
			const msg =
				err instanceof Error
					? err.message
					: "Something went wrong talking to the API.";
			const errorMessage: ChatMessage = {
				id: Date.now(),
				role: "error",
				content: msg,
			};
			setMessages((prev) => [...prev, errorMessage]);
		} finally {
			setLoading(false);
		}
	}

	return (
		<main className="app-root">
			<header>
				<h1>UK GVT RAG</h1>
				<p className="subtitle">Ask a question about the seeded GOV.UK docs.</p>
			</header>

			<section className="chat-shell">
				<div className="chat-messages">
					{messages.map((msg) => {
						// Special system message: list seeded docs + example questions
						if (msg.role === "assistant" && msg.kind === "system") {
							return (
								<div
									key={msg.id}
									className="chat-message-row chat-message-row-assistant"
                                >
									<div className="chat-bubble">
										<strong>Assistant</strong>
										<p>
											You can ask questions about the following seeded GOV.UK
											documents:
										</p>
										<ul className="seeded-list">
											<li>National Risk Register 2025</li>
											<li>The UK Government Resilience Framework</li>
											<li>British Energy Security Strategy</li>
											<li>Government Cyber Security Strategy 2022–2030</li>
											<li>Net Zero Strategy: Build Back Greener</li>
										</ul>
										<p className="seeded-examples-label">Example questions:</p>
										<ul className="seeded-list">
											<li>
												How does the National Risk Register describe the main
												impacts of coastal flooding?
											</li>
											<li>
												What are the key principles of resilience in the UK
												Government Resilience Framework?
											</li>
											<li>
												What are the main goals of the Net Zero Strategy: Build
												Back Greener?
											</li>
										</ul>
									</div>
								</div>
							);
						}

						const rowClass = `chat-message-row chat-message-row-${msg.role}`;

						const bubbleClasses = ["chat-bubble"];
						if (msg.role === "assistant" && msg.kind === "chunks") {
							bubbleClasses.push("chat-bubble-chunks");
						}
						if (msg.role === "assistant" && msg.kind === "answer") {
							bubbleClasses.push("chat-bubble-answer");
						}
						if (msg.role === "error") {
							bubbleClasses.push("chat-bubble-error");
						}
						const bubbleClassName = bubbleClasses.join(" ");

						return (
							<div key={msg.id} className={rowClass}>
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

										{msg.citations && msg.citations.length > 0 && (
											<div className="chat-citations">
												<p className="chat-citations-label">Sources:</p>
												<ul>
													{msg.citations.map((c, idx) => (
														<li key={`${idx}-${c.title}-${c.url}`}>
															{c.title || "Unknown source"}
															{c.url && (
																<>
																	{" "}
																	(<a
																		href={buildSourceUrl(c.url, c.page)}
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

								{msg.role === "assistant" && msg.kind === "chunks" && msg.chunks && (
									<div className={bubbleClassName}>
										<strong>Retrieved chunks</strong>
										<ul className="chunk-list">
											{msg.chunks.map((c, idx) => (
												<li
													key={`${idx}-${c.title}-${c.url}`}
													className="chunk-item"
												>
													<div className="chunk-meta">
														<span className="chunk-title">
															{c.title || `Chunk ${idx + 1}`}
														</span>
														{c.url && (
															<a
																href={buildSourceUrl(c.url, c.page)}
																target="_blank"
																rel="noreferrer"
															>
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
