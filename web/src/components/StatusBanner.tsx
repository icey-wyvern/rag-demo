import type { CorpusStatus } from "../types";

interface StatusBannerProps {
  status: CorpusStatus | null;
  statusError: string;
}

function statusMessage(status: CorpusStatus | null): string {
  if (!status) {
    return "Checking corpus status.";
  }
  if (status.status === "ready") {
    const count = status.versioned?.sources ?? status.documents_total;
    return `Versioned corpus ready. ${count} ${count === 1 ? "source" : "sources"} and ${status.versioned?.current_passages ?? 0} current passages available.`;
  }
  if (status.status === "running") {
    return `Versioned corpus setup in progress: ${status.versioned?.current_passages ?? 0} current passages, ${status.versioned?.embedded_passages ?? 0} embeddings.${status.versioned?.lexical_ready ? " Lexical retrieval is available." : ""}`;
  }
  if (status.status === "error") {
    return status.last_error || "Versioned corpus is not ready for default hybrid queries.";
  }
  return "Versioned corpus is not ready yet.";
}

export function StatusBanner({ status, statusError }: StatusBannerProps) {
  const statusClass =
    status?.status === "ready"
      ? "status-banner-ready"
      : status?.status === "error" || statusError
        ? "status-banner-error"
        : "status-banner-pending";

  return (
    <section className={`status-banner ${statusClass}`} role="status">
      <strong>Corpus status</strong>
      <p>{statusError || statusMessage(status)}</p>
      {!statusError && status?.status !== "ready" && (
        <p className="status-banner-note">
          Prepare the versioned guidance corpus and embeddings. This status updates automatically.
        </p>
      )}
    </section>
  );
}
