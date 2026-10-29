import type { ScoredId } from "./bm25.js";

/**
 * Cosine similarity between two vectors.
 *
 * Returns 0 when either vector has no magnitude, rather than NaN — a zero
 * vector is a legitimate thing to get back from an embedder for empty input,
 * and a NaN propagating into a ranking silently destroys it.
 */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) {
    throw new Error(`vector length mismatch: ${a.length} vs ${b.length}`);
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dot / magnitude;
}

/**
 * Exhaustive nearest-neighbour search over vectors held in memory.
 *
 * This is a linear scan. For the corpus sizes this service is meant for —
 * a product's documentation, a handbook, a few thousand chunks — a scan is
 * both fast enough and exactly correct, where an approximate index trades
 * recall for a speedup nobody here would notice. Past roughly 10^5 vectors,
 * swap this for pgvector or a dedicated index; the `Retriever` only needs
 * `search`.
 */
export class InMemoryVectorIndex {
  readonly #vectors = new Map<string, number[]>();
  #dimensions: number | null = null;

  get size(): number {
    return this.#vectors.size;
  }

  get dimensions(): number | null {
    return this.#dimensions;
  }

  add(id: string, vector: number[]): void {
    if (this.#dimensions === null) {
      this.#dimensions = vector.length;
    } else if (vector.length !== this.#dimensions) {
      // Mixing dimensions means two different embedding models got into one
      // index, and every score across them is meaningless.
      throw new Error(
        `vector for "${id}" has ${vector.length} dimensions, index holds ${this.#dimensions}`,
      );
    }
    this.#vectors.set(id, vector);
  }

  remove(id: string): void {
    this.#vectors.delete(id);
  }

  search(query: readonly number[], limit = 10): ScoredId[] {
    if (this.#vectors.size === 0) return [];

    const scored: ScoredId[] = [];
    for (const [id, vector] of this.#vectors) {
      scored.push({ id, score: cosineSimilarity(query, vector) });
    }

    return scored
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
      .slice(0, limit);
  }
}
