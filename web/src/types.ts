export type Mode = "chunks" | "generate";
export type RetrievalMode = "hybrid" | "semantic" | "lexical";
export type QueryStatus = "answered" | "abstained" | "retrieved";

export interface Provenance {
  source_key?: string;
  snapshot_id?: string;
  snapshot_content_sha256?: string;
  observed_at?: string;
  source_updated_at?: string | null;
  passage_id?: string;
  passage_ordinal?: number;
  passage_sha256?: string;
}

export interface Chunk extends Provenance {
  title: string;
  url: string;
  snippet: string;
  text?: string;
  page?: number | null;
}

export interface Citation extends Provenance {
  title: string;
  url: string;
  page?: number | null;
}

export interface ClaimCitation extends Citation {
  evidence_id: string;
  supporting_quote: string;
}

export interface Claim {
  text: string;
  citations: ClaimCitation[];
}

export interface RetrievalMetadata {
  retrieval_mode: RetrievalMode;
  k: number;
  as_of: string | null;
  evidence_count: number;
}

export interface QueryResponse {
  status: QueryStatus;
  answer: string;
  abstention_reason?: string | null;
  citations: Citation[];
  sources?: Citation[];
  claims: Claim[];
  chunks: Chunk[];
  retrieval?: RetrievalMetadata;
}

export interface CorpusStatus {
  status: "pending" | "running" | "ready" | "error";
  ingest_version: number;
  documents_total: number;
  documents_ready: number;
  last_error?: string | null;
  is_ready: boolean;
  versioned?: {
    schema_ready: boolean;
    sources: number;
    snapshots: number;
    current_passages: number;
    embedded_passages: number;
    lexical_ready: boolean;
    semantic_ready: boolean;
    provider_configured: boolean;
  };
}

export type MessageKind = "system" | "answer" | "chunks";

export interface ChatMessage {
  id: number;
  role: "user" | "assistant" | "error";
  kind?: MessageKind;
  content?: string;
  result?: QueryResponse;
}
