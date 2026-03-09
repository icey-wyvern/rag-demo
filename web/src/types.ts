export type Mode = "chunks" | "generate";

export interface Chunk {
  title: string;
  url: string;
  snippet: string;
  page?: number;
}

export interface Citation {
  title: string;
  url: string;
  page?: number;
}

export interface QueryResponse {
  answer?: string;
  citations?: Citation[];
  chunks: Chunk[];
}

export interface CorpusStatus {
  status: "pending" | "running" | "ready" | "error";
  ingest_version: number;
  documents_total: number;
  documents_ready: number;
  last_error?: string | null;
  is_ready: boolean;
}

export type MessageKind = "system" | "answer" | "chunks";

export interface ChatMessage {
  id: number;
  role: "user" | "assistant" | "error";
  kind?: MessageKind;
  content?: string;
  chunks?: Chunk[];
  citations?: Citation[];
}
