/**
 * BM25 lexical search.
 *
 * Dense vectors get the attention, but lexical search is what finds the exact
 * identifier, error code or flag a user pasted in — the queries where an
 * embedding model's "semantic similarity" is actively unhelpful. The two are
 * complements, which is why `Retriever` runs both and fuses the rankings.
 *
 * This implementation has no stemmer. A stemmer is a per-language asset and a
 * bad one ("universe" and "university" collapsing) costs more precision than
 * it buys recall, so it is left out rather than half-done.
 */

export interface Bm25Options {
  /** Term-frequency saturation. Default 1.5. */
  k1?: number;
  /** Length normalization, 0 = none, 1 = full. Default 0.75. */
  b?: number;
}

export interface ScoredId {
  id: string;
  score: number;
}

/** Common words that match everything and therefore distinguish nothing. */
const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
  "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "of",
  "on", "or", "she", "so", "that", "the", "their", "them", "then", "there",
  "these", "they", "this", "to", "was", "were", "what", "when", "which", "who",
  "will", "with", "you", "your",
]);

/**
 * Lowercases and splits on anything that is not a letter, digit, underscore,
 * hyphen or dot — so `max_tokens`, `claude-opus-5-5` and `v1.2.3` survive as
 * single terms instead of shattering into noise.
 */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^\p{L}\p{N}_.-]+/u)
    .map((term) => term.replace(/^[.-]+|[.-]+$/g, ""))
    .filter((term) => term.length > 1 && !STOPWORDS.has(term));
}

interface Posting {
  id: string;
  frequency: number;
}

export class Bm25Index {
  readonly #k1: number;
  readonly #b: number;
  /** term -> postings */
  readonly #postings = new Map<string, Posting[]>();
  readonly #lengths = new Map<string, number>();
  #totalLength = 0;

  constructor(options: Bm25Options = {}) {
    this.#k1 = options.k1 ?? 1.5;
    this.#b = options.b ?? 0.75;
  }

  get size(): number {
    return this.#lengths.size;
  }

  add(id: string, text: string): void {
    if (this.#lengths.has(id)) this.remove(id);

    const terms = tokenize(text);
    const frequencies = new Map<string, number>();
    for (const term of terms) {
      frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
    }

    for (const [term, frequency] of frequencies) {
      const list = this.#postings.get(term);
      if (list) list.push({ id, frequency });
      else this.#postings.set(term, [{ id, frequency }]);
    }

    this.#lengths.set(id, terms.length);
    this.#totalLength += terms.length;
  }

  remove(id: string): void {
    const length = this.#lengths.get(id);
    if (length === undefined) return;

    for (const [term, list] of this.#postings) {
      const next = list.filter((posting) => posting.id !== id);
      if (next.length === 0) this.#postings.delete(term);
      else this.#postings.set(term, next);
    }

    this.#lengths.delete(id);
    this.#totalLength -= length;
  }

  #averageLength(): number {
    return this.#lengths.size === 0 ? 0 : this.#totalLength / this.#lengths.size;
  }

  /**
   * Scores documents against a query.
   *
   * Uses the standard BM25 idf with the +1 inside the log, which keeps the
   * score non-negative — the textbook form goes negative for terms appearing
   * in more than half the corpus, which on a small corpus means a common word
   * can actively push a document down.
   */
  search(query: string, limit = 10): ScoredId[] {
    const terms = tokenize(query);
    if (terms.length === 0 || this.#lengths.size === 0) return [];

    const documentCount = this.#lengths.size;
    const averageLength = this.#averageLength();
    const scores = new Map<string, number>();

    for (const term of new Set(terms)) {
      const list = this.#postings.get(term);
      if (!list) continue;

      const idf = Math.log(1 + (documentCount - list.length + 0.5) / (list.length + 0.5));

      for (const { id, frequency } of list) {
        const length = this.#lengths.get(id) ?? 0;
        const denominator =
          frequency +
          this.#k1 * (1 - this.#b + (this.#b * length) / (averageLength || 1));
        const contribution = idf * ((frequency * (this.#k1 + 1)) / denominator);
        scores.set(id, (scores.get(id) ?? 0) + contribution);
      }
    }

    return [...scores.entries()]
      .map(([id, score]) => ({ id, score }))
      // Ties broken by id so results are stable between runs.
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
      .slice(0, limit);
  }
}
