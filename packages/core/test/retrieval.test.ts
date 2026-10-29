import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  Bm25Index,
  chunkMarkdown,
  cosineSimilarity,
  indexableText,
  InMemoryVectorIndex,
  reciprocalRankFusion,
  tokenize,
} from "../src/index.js";

describe("tokenize", () => {
  it("keeps identifiers whole instead of shattering them", () => {
    assert.deepEqual(tokenize("Set max_tokens on claude-opus-5-5 (v1.2.3)"), [
      "set",
      "max_tokens",
      "claude-opus-5-5",
      "v1.2.3",
    ]);
  });

  it("drops stopwords and single characters", () => {
    assert.deepEqual(tokenize("the a of X retrieval"), ["retrieval"]);
  });

  it("strips leading and trailing punctuation", () => {
    assert.deepEqual(tokenize("...rate-limits."), ["rate-limits"]);
  });
});

describe("chunkMarkdown", () => {
  const doc = [
    "# API",
    "",
    "The API is stateless.",
    "",
    "## Errors",
    "",
    "Errors come back as JSON.",
    "",
    "### Rate limits",
    "",
    "Requests are capped per minute. ".repeat(40),
  ].join("\n");

  it("carries the heading path onto each chunk", () => {
    const chunks = chunkMarkdown("doc1", doc);
    const deepest = chunks.at(-1)!;
    assert.deepEqual(deepest.headings, ["API", "Errors", "Rate limits"]);
  });

  it("makes the heading path part of what gets indexed", () => {
    const [first] = chunkMarkdown("doc1", doc);
    assert.match(indexableText(first!), /^API/);
  });

  it("gives chunks resolvable ids", () => {
    const chunks = chunkMarkdown("doc1", doc);
    assert.equal(chunks[0]?.id, "doc1#0");
    assert.deepEqual(
      chunks.map((chunk) => chunk.index),
      chunks.map((_, i) => i),
    );
    assert.ok(chunks.every((chunk) => chunk.documentId === "doc1"));
  });

  it("splits a long section but keeps short ones whole", () => {
    const chunks = chunkMarkdown("doc1", doc, { maxChars: 300, minChars: 0 });
    const rateLimit = chunks.filter((chunk) => chunk.headings.includes("Rate limits"));
    assert.ok(rateLimit.length > 1, "the long section is split");
    assert.ok(
      rateLimit.every((chunk) => chunk.text.length <= 600),
      "no chunk runs far past the ceiling",
    );
  });

  it("merges a stub section into its neighbour", () => {
    const stubby = ["# One", "", "Short.", "", "## Two", "", "Also short."].join("\n");
    const chunks = chunkMarkdown("doc1", stubby, { minChars: 200 });
    assert.equal(chunks.length, 1);
    assert.match(chunks[0]!.text, /Short\./);
    assert.match(chunks[0]!.text, /Also short\./);
  });

  it("returns nothing for an empty document", () => {
    assert.deepEqual(chunkMarkdown("empty", "   \n\n  "), []);
  });
});

describe("Bm25Index", () => {
  const index = new Bm25Index();
  index.add("a", "The agent loop calls tools and feeds results back.");
  index.add("b", "Context budgeting clears old tool output from the transcript.");
  index.add("c", "Rate limits are enforced per minute on every endpoint.");

  it("ranks the document that actually matches first", () => {
    const [top] = index.search("rate limits per minute");
    assert.equal(top?.id, "c");
  });

  it("returns nothing for a query of only stopwords", () => {
    assert.deepEqual(index.search("the a of"), []);
  });

  it("never scores a document negatively", () => {
    // The textbook idf goes negative for terms in over half the corpus, which
    // on a small corpus lets a common word push a document down the ranking.
    const common = new Bm25Index();
    common.add("x", "tool tool tool");
    common.add("y", "tool something else");
    common.add("z", "tool another thing");
    for (const hit of common.search("tool")) {
      assert.ok(hit.score >= 0, `${hit.id} scored ${hit.score}`);
    }
  });

  it("forgets a removed document", () => {
    const mutable = new Bm25Index();
    mutable.add("a", "quokka");
    assert.equal(mutable.search("quokka").length, 1);
    mutable.remove("a");
    assert.equal(mutable.search("quokka").length, 0);
    assert.equal(mutable.size, 0);
  });

  it("replaces rather than duplicates on re-add", () => {
    const mutable = new Bm25Index();
    mutable.add("a", "first text");
    mutable.add("a", "second text");
    assert.equal(mutable.size, 1);
    assert.equal(mutable.search("first").length, 0);
    assert.equal(mutable.search("second").length, 1);
  });

  it("orders ties deterministically", () => {
    const ties = new Bm25Index();
    ties.add("b", "identical text here");
    ties.add("a", "identical text here");
    assert.deepEqual(
      ties.search("identical").map((hit) => hit.id),
      ["a", "b"],
    );
  });
});

describe("cosineSimilarity", () => {
  it("is 1 for identical direction and 0 for orthogonal", () => {
    assert.equal(cosineSimilarity([1, 0], [2, 0]), 1);
    assert.equal(cosineSimilarity([1, 0], [0, 1]), 0);
  });

  it("returns 0 rather than NaN for a zero vector", () => {
    assert.equal(cosineSimilarity([0, 0], [1, 1]), 0);
  });

  it("refuses mismatched lengths", () => {
    assert.throws(() => cosineSimilarity([1, 2], [1]), /length mismatch/);
  });
});

describe("InMemoryVectorIndex", () => {
  it("returns the nearest vectors in order", () => {
    const index = new InMemoryVectorIndex();
    index.add("near", [1, 0.1]);
    index.add("far", [-1, 0]);
    index.add("mid", [0.5, 0.5]);
    assert.deepEqual(
      index.search([1, 0], 2).map((hit) => hit.id),
      ["near", "mid"],
    );
  });

  it("rejects a vector of the wrong dimension", () => {
    const index = new InMemoryVectorIndex();
    index.add("a", [1, 2, 3]);
    assert.throws(() => index.add("b", [1, 2]), /dimensions/);
  });

  it("is empty-safe", () => {
    assert.deepEqual(new InMemoryVectorIndex().search([1, 2]), []);
  });
});

describe("reciprocalRankFusion", () => {
  it("rewards agreement across lists over one list's top hit", () => {
    const lexical = [
      { id: "solo", score: 99 },
      { id: "agreed", score: 2 },
    ];
    const dense = [
      { id: "agreed", score: 0.9 },
      { id: "other", score: 0.8 },
    ];
    const [top] = reciprocalRankFusion([lexical, dense]);
    assert.equal(top?.id, "agreed");
  });

  it("ignores the magnitude of incomparable scores", () => {
    // BM25 sums are unbounded; cosine lives in [-1, 1]. Only rank is used.
    const a = reciprocalRankFusion([[{ id: "x", score: 1000 }], [{ id: "y", score: 0.1 }]]);
    const b = reciprocalRankFusion([[{ id: "x", score: 0.001 }], [{ id: "y", score: 42 }]]);
    assert.deepEqual(
      a.map((entry) => entry.id),
      b.map((entry) => entry.id),
    );
  });

  it("reports where each result ranked", () => {
    const fused = reciprocalRankFusion([
      [{ id: "a", score: 1 }],
      [
        { id: "b", score: 1 },
        { id: "a", score: 0.5 },
      ],
    ]);
    const a = fused.find((entry) => entry.id === "a");
    assert.deepEqual(a?.ranks, [1, 2]);
    const b = fused.find((entry) => entry.id === "b");
    assert.deepEqual(b?.ranks, [null, 1]);
  });

  it("honours per-list weights", () => {
    const lists = [[{ id: "lex", score: 1 }], [{ id: "vec", score: 1 }]];
    assert.equal(reciprocalRankFusion(lists, { weights: [3, 1] })[0]?.id, "lex");
    assert.equal(reciprocalRankFusion(lists, { weights: [1, 3] })[0]?.id, "vec");
  });

  it("rejects a weight/list mismatch", () => {
    assert.throws(
      () => reciprocalRankFusion([[], []], { weights: [1] }),
      /expected 2 weights/,
    );
  });
});
