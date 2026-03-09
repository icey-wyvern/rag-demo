import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

describe("App", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows pending corpus status from the API", async () => {
    vi.spyOn(window, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          status: "running",
          ingest_version: 1,
          documents_total: 5,
          documents_ready: 2,
          last_error: null,
          is_ready: false,
        }),
        { status: 200 },
      ),
    );

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Corpus ingestion in progress/i)).toBeInTheDocument();
    });
  });
});
