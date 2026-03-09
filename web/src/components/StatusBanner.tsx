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
    return `Corpus ready. Indexed ${status.documents_ready} of ${status.documents_total} documents.`;
  }
  if (status.status === "running") {
    return `Corpus ingestion in progress: ${status.documents_ready} of ${status.documents_total} documents indexed.`;
  }
  if (status.status === "error") {
    return status.last_error || "Corpus ingestion failed.";
  }
  return "Corpus has not been built yet. Start the ingest service to prepare retrieval data.";
}

export function StatusBanner({ status, statusError }: StatusBannerProps) {
  const statusClass =
    status?.status === "ready"
      ? "status-banner-ready"
      : status?.status === "error" || statusError
        ? "status-banner-error"
        : "status-banner-pending";

  return (
    <section className={`status-banner ${statusClass}`}>
      <strong>Corpus status</strong>
      <p>{statusError || statusMessage(status)}</p>
      {!statusError && status?.status !== "ready" && (
        <p className="status-banner-note">
          Run `docker compose up --build` and wait for the `ingest` service to finish. The page will update automatically.
        </p>
      )}
    </section>
  );
}
