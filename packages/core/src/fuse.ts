import type { ScoredId } from "./bm25.js";

export interface FusionOptions {
  /**
   * Rank-smoothing constant. Larger flattens the curve and lets agreement
   * across lists matter more than a single list's top hit. 60 is the value
   * from the original paper and is a reasonable default.
   */
  k?: number;
  /** Per-list weights, aligned with the rankings array. Default: all 1. */
  weights?: number[];
}

export interface FusedResult extends ScoredId {
  /** 1-based rank in each input list, or null where it did not appear. */
  ranks: Array<number | null>;
}

/**
 * Reciprocal rank fusion.
 *
 * Combining a BM25 ranking with a vector ranking by score is a category error:
 * BM25 scores are unbounded sums of idf terms and cosine scores live in
 * [-1, 1], so any normalization you pick is arbitrary and corpus-dependent.
 * RRF throws the scores away and uses only rank, which makes the combination
 * scale-free — a document ranked well by both retrievers beats one ranked
 * brilliantly by one and not at all by the other.
 */
export function reciprocalRankFusion(
  rankings: ReadonlyArray<readonly ScoredId[]>,
  options: FusionOptions = {},
): FusedResult[] {
  const k = options.k ?? 60;
  const weights = options.weights ?? rankings.map(() => 1);
  if (weights.length !== rankings.length) {
    throw new Error(
      `expected ${rankings.length} weights, got ${weights.length}`,
    );
  }

  const scores = new Map<string, number>();
  const ranks = new Map<string, Array<number | null>>();

  rankings.forEach((ranking, listIndex) => {
    ranking.forEach((entry, position) => {
      const rank = position + 1;
      scores.set(
        entry.id,
        (scores.get(entry.id) ?? 0) + (weights[listIndex] ?? 1) / (k + rank),
      );
      const existing = ranks.get(entry.id) ?? rankings.map(() => null);
      existing[listIndex] = rank;
      ranks.set(entry.id, existing);
    });
  });

  return [...scores.entries()]
    .map(([id, score]) => ({
      id,
      score,
      ranks: ranks.get(id) ?? rankings.map(() => null),
    }))
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
}
