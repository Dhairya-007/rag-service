import { Bm25Index, type Bm25Options, type ScoredId } from "./bm25.js";
import { chunkMarkdown, indexableText, type Chunk, type ChunkOptions } from "./chunk.js";
import type { Embedder } from "./embedder.js";
import { reciprocalRankFusion } from "./fuse.js";
import { InMemoryVectorIndex } from "./vector.js";

export interface Document {
  id: string;
  title: string;
  text: string;
  /** Where this came from, for citations. */
  source?: string;
}

export interface RetrievedChunk {
  chunk: Chunk;
  document: Pick<Document, "id" | "title" | "source">;
  /** Fused score. Comparable within one result set, not across queries. */
  score: number;
  /** Rank in the lexical and dense lists, where each ran. */
  ranks: { lexical: number | null; dense: number | null };
}

export type RetrievalMode = "hybrid" | "lexical" | "dense";

export interface SearchOptions {
  limit?: number;
  mode?: RetrievalMode;
  /** How deep to go in each retriever before fusing. Default `limit * 4`. */
  candidates?: number;
}

export interface RetrieverOptions {
  embedder?: Embedder;
  chunking?: ChunkOptions;
  bm25?: Bm25Options;
}

/**
 * Ingests documents and answers queries against them.
 *
 * Retrieval is hybrid by default: BM25 for the queries that are really keyword
 * lookups, dense vectors for the ones that are paraphrases, and reciprocal
 * rank fusion to combine them without pretending their scores are comparable.
 * With no embedder configured, it runs lexical-only and says so.
 *
 * Storage is in memory. The indexes behind `search` are the only coupling, so
 * moving to pgvector means replacing two calls, not restructuring this class —
 * but shipping an untested Postgres adapter would be worse than shipping none,
 * so it is not in the box.
 */
export class Retriever {
  readonly #lexical: Bm25Index;
  readonly #dense = new InMemoryVectorIndex();
  readonly #chunks = new Map<string, Chunk>();
  readonly #documents = new Map<string, Document>();
  readonly #embedder: Embedder | undefined;
  readonly #chunking: ChunkOptions;

  constructor(options: RetrieverOptions = {}) {
    this.#embedder = options.embedder;
    this.#chunking = options.chunking ?? {};
    this.#lexical = new Bm25Index(options.bm25);
  }

  get hasEmbedder(): boolean {
    return this.#embedder !== undefined;
  }

  get documentCount(): number {
    return this.#documents.size;
  }

  get chunkCount(): number {
    return this.#chunks.size;
  }

  documents(): Document[] {
    return [...this.#documents.values()];
  }

  getChunk(id: string): Chunk | undefined {
    return this.#chunks.get(id);
  }

  /** Re-ingesting the same id replaces the document and all of its chunks. */
  async ingest(document: Document): Promise<Chunk[]> {
    this.remove(document.id);

    const chunks = chunkMarkdown(document.id, document.text, this.#chunking);
    if (chunks.length === 0) {
      this.#documents.set(document.id, document);
      return [];
    }

    for (const chunk of chunks) {
      this.#chunks.set(chunk.id, chunk);
      this.#lexical.add(chunk.id, indexableText(chunk));
    }

    if (this.#embedder) {
      const vectors = await this.#embedder.embed(
        chunks.map((chunk) => indexableText(chunk)),
        "document",
      );
      chunks.forEach((chunk, i) => {
        const vector = vectors[i];
        if (vector) this.#dense.add(chunk.id, vector);
      });
    }

    this.#documents.set(document.id, document);
    return chunks;
  }

  remove(documentId: string): void {
    for (const [id, chunk] of this.#chunks) {
      if (chunk.documentId !== documentId) continue;
      this.#chunks.delete(id);
      this.#lexical.remove(id);
      this.#dense.remove(id);
    }
    this.#documents.delete(documentId);
  }

  async search(query: string, options: SearchOptions = {}): Promise<RetrievedChunk[]> {
    const limit = options.limit ?? 8;
    const candidates = options.candidates ?? limit * 4;
    const requested = options.mode ?? "hybrid";

    // Asking for dense retrieval without an embedder is a configuration
    // mistake worth surfacing, not silently serving lexical results for.
    if (requested === "dense" && !this.#embedder) {
      throw new Error("dense retrieval requires an embedder");
    }
    const mode: RetrievalMode =
      requested === "hybrid" && !this.#embedder ? "lexical" : requested;

    const lexical: ScoredId[] =
      mode === "dense" ? [] : this.#lexical.search(query, candidates);

    let dense: ScoredId[] = [];
    if (mode !== "lexical" && this.#embedder) {
      const [vector] = await this.#embedder.embed([query], "query");
      if (vector) dense = this.#dense.search(vector, candidates);
    }

    const lists = [lexical, dense];
    const fused = reciprocalRankFusion(lists).slice(0, limit);

    const results: RetrievedChunk[] = [];
    for (const entry of fused) {
      const chunk = this.#chunks.get(entry.id);
      if (!chunk) continue;
      const document = this.#documents.get(chunk.documentId);
      if (!document) continue;

      results.push({
        chunk,
        document: {
          id: document.id,
          title: document.title,
          ...(document.source ? { source: document.source } : {}),
        },
        score: entry.score,
        ranks: { lexical: entry.ranks[0] ?? null, dense: entry.ranks[1] ?? null },
      });
    }

    return results;
  }
}

/** Renders results as the context block an agent reads. */
export function formatContext(results: readonly RetrievedChunk[]): string {
  if (results.length === 0) return "No matching passages were found.";

  return results
    .map((result) => {
      const breadcrumb =
        result.chunk.headings.length > 0
          ? ` — ${result.chunk.headings.join(" > ")}`
          : "";
      return [
        `[${result.chunk.id}] ${result.document.title}${breadcrumb}`,
        result.chunk.text,
      ].join("\n");
    })
    .join("\n\n---\n\n");
}
