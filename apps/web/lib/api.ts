export const API_URL =
  process.env["NEXT_PUBLIC_API_URL"] ?? "http://localhost:3001";

export interface DocumentSummary {
  id: string;
  title: string;
  source?: string;
}

export interface SearchHit {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  headings: string[];
  text: string;
  score: number;
  ranks: { lexical: number | null; dense: number | null };
}

export interface Citation {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  headings: string[];
  source?: string;
  excerpt: string;
}

export interface AskResult {
  answer: string;
  citations: Citation[];
  steps: number;
  verified: boolean | null;
  usage: { inputTokens: number; outputTokens: number };
  searches: string[];
}

export interface Health {
  status: string;
  corpus: { documents: number; chunks: number; hybrid: boolean };
}

/**
 * Thrown for any non-2xx response, carrying the API's own message.
 *
 * The API returns useful errors — which DTO field failed, that credentials are
 * missing — and swallowing them into "Something went wrong" would waste the
 * most helpful thing the server said.
 */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_URL}. Is it running?`);
  }

  if (response.status === 204) return undefined as T;

  const body = (await response.json().catch(() => null)) as
    | { message?: string | string[] }
    | null;

  if (!response.ok) {
    const message = Array.isArray(body?.message)
      ? body.message.join("; ")
      : (body?.message ?? response.statusText);
    throw new ApiError(response.status, message);
  }

  return body as T;
}

export const api = {
  health: () => call<Health>("/health"),
  listDocuments: () => call<{ documents: DocumentSummary[] }>("/documents"),
  ingest: (document: { id: string; title: string; text: string; source?: string }) =>
    call<{ id: string; chunks: number }>("/documents", {
      method: "POST",
      body: JSON.stringify(document),
    }),
  remove: (id: string) =>
    call<void>(`/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  search: (query: string, mode: "hybrid" | "lexical" = "hybrid") =>
    call<{ results: SearchHit[] }>("/search", {
      method: "POST",
      body: JSON.stringify({ query, mode }),
    }),
  ask: (question: string) =>
    call<AskResult>("/ask", {
      method: "POST",
      body: JSON.stringify({ question }),
    }),
};
