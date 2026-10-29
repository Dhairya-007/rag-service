/**
 * Dense embeddings.
 *
 * The Claude API has no embeddings endpoint — Claude is the generation half of
 * a RAG system, not the retrieval half. So this is an interface, with the
 * provider left to the caller. Voyage is Anthropic's documented embedding
 * partner and gets an adapter below; anything that turns text into vectors
 * fits the same shape.
 *
 * Dense retrieval is optional throughout this package. Without an embedder the
 * `Retriever` runs BM25 alone, which is a real retrieval system, not a
 * degraded one — and it keeps the whole project runnable and testable with no
 * second API key.
 */

export interface Embedder {
  /** Vector length this embedder produces. */
  readonly dimensions: number;
  /** Embeds a batch. Must return one vector per input, in order. */
  embed(texts: string[], kind: "document" | "query"): Promise<number[][]>;
}

export interface VoyageEmbedderOptions {
  apiKey?: string;
  /** Defaults to `voyage-3`. */
  model?: string;
  dimensions?: number;
  baseUrl?: string;
  /** Texts per request. Default 64. */
  batchSize?: number;
  fetch?: typeof globalThis.fetch;
}

interface VoyageResponse {
  data?: Array<{ embedding: number[]; index: number }>;
  detail?: string;
}

/**
 * Voyage AI adapter.
 *
 * Note the `kind` argument: Voyage wants to know whether it is embedding a
 * stored passage or a search query, and embedding a query as a document
 * measurably costs recall. The interface carries it for that reason.
 */
export class VoyageEmbedder implements Embedder {
  readonly dimensions: number;
  readonly #apiKey: string;
  readonly #model: string;
  readonly #baseUrl: string;
  readonly #batchSize: number;
  readonly #fetch: typeof globalThis.fetch;

  constructor(options: VoyageEmbedderOptions = {}) {
    const apiKey = options.apiKey ?? process.env["VOYAGE_API_KEY"];
    if (!apiKey) {
      throw new Error(
        "VoyageEmbedder needs an API key (pass `apiKey` or set VOYAGE_API_KEY)",
      );
    }
    this.#apiKey = apiKey;
    this.#model = options.model ?? "voyage-3";
    this.dimensions = options.dimensions ?? 1024;
    this.#baseUrl = options.baseUrl ?? "https://api.voyageai.com/v1";
    this.#batchSize = options.batchSize ?? 64;
    this.#fetch = options.fetch ?? globalThis.fetch;
  }

  async embed(texts: string[], kind: "document" | "query"): Promise<number[][]> {
    const out: number[][] = [];

    for (let offset = 0; offset < texts.length; offset += this.#batchSize) {
      const batch = texts.slice(offset, offset + this.#batchSize);
      const response = await this.#fetch(`${this.#baseUrl}/embeddings`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.#apiKey}`,
        },
        body: JSON.stringify({
          model: this.#model,
          input: batch,
          input_type: kind,
        }),
      });

      const body = (await response.json()) as VoyageResponse;
      if (!response.ok) {
        throw new Error(
          `Voyage embeddings failed (${response.status}): ${body.detail ?? "unknown error"}`,
        );
      }
      if (!body.data || body.data.length !== batch.length) {
        throw new Error(
          `Voyage returned ${body.data?.length ?? 0} embeddings for ${batch.length} inputs`,
        );
      }

      // The API does not promise response order, and a silently misaligned
      // batch is the kind of bug that shows up as "retrieval got worse".
      const ordered = [...body.data].sort((a, b) => a.index - b.index);
      for (const entry of ordered) out.push(entry.embedding);
    }

    return out;
  }
}
