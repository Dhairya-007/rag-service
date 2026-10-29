import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { formatContext, Retriever, tokenize, type Embedder } from "../src/index.js";

/**
 * A deterministic stand-in for a real embedding model: a bag-of-words vector
 * over a fixed vocabulary. It is a poor embedder and a good test double —
 * synonyms do not help it, but it is stable, offline and free.
 */
class BagOfWordsEmbedder implements Embedder {
  readonly dimensions: number;
  readonly #vocabulary: string[];
  /** Records what it was asked, so tests can assert on query vs document. */
  readonly calls: Array<{ count: number; kind: "document" | "query" }> = [];

  constructor(vocabulary: string[]) {
    this.#vocabulary = vocabulary;
    this.dimensions = vocabulary.length;
  }

  async embed(texts: string[], kind: "document" | "query"): Promise<number[][]> {
    this.calls.push({ count: texts.length, kind });
    return texts.map((text) => {
      const terms = new Set(tokenize(text));
      return this.#vocabulary.map((word) => (terms.has(word) ? 1 : 0));
    });
  }
}

const handbook = {
  id: "handbook",
  title: "Engineering Handbook",
  source: "docs/handbook.md",
  text: [
    "# Deployments",
    "",
    "Every service ships through the pipeline. Releases are cut on Tuesdays.",
    "",
    "## Rollbacks",
    "",
    "A rollback reverts to the previous image tag and takes about four minutes.",
    "",
    "# Oncall",
    "",
    "The oncall rotation is weekly. Pages route to the primary first.",
  ].join("\n"),
};

const runbook = {
  id: "runbook",
  title: "Database Runbook",
  text: [
    "# Failover",
    "",
    "Promote the replica, then repoint the connection string.",
    "",
    "# Backups",
    "",
    "Snapshots run nightly and are retained for thirty days.",
  ].join("\n"),
};

describe("Retriever ingestion", () => {
  it("indexes a document into chunks", async () => {
    const retriever = new Retriever();
    const chunks = await retriever.ingest(handbook);

    assert.ok(chunks.length > 0);
    assert.equal(retriever.documentCount, 1);
    assert.equal(retriever.chunkCount, chunks.length);
    assert.equal(retriever.hasEmbedder, false);
  });

  it("replaces a document rather than duplicating it", async () => {
    const retriever = new Retriever();
    await retriever.ingest(handbook);
    const before = retriever.chunkCount;

    await retriever.ingest({ ...handbook, text: "# Deployments\n\nNow much shorter." });

    assert.equal(retriever.documentCount, 1);
    assert.ok(retriever.chunkCount < before, "stale chunks are gone");
    const hits = await retriever.search("Tuesdays");
    assert.equal(hits.length, 0, "the old text is no longer retrievable");
  });

  it("removes a document and its chunks", async () => {
    const retriever = new Retriever();
    await retriever.ingest(handbook);
    await retriever.ingest(runbook);

    retriever.remove("handbook");

    assert.equal(retriever.documentCount, 1);
    assert.ok(retriever.documents().every((doc) => doc.id === "runbook"));
    const hits = await retriever.search("rollback image tag");
    assert.ok(hits.every((hit) => hit.document.id === "runbook"));
  });

  it("handles a document with no content", async () => {
    const retriever = new Retriever();
    const chunks = await retriever.ingest({ id: "blank", title: "Blank", text: "" });
    assert.deepEqual(chunks, []);
    assert.equal(retriever.documentCount, 1);
  });
});

describe("Retriever search", () => {
  it("finds the passage that answers the question", async () => {
    const retriever = new Retriever();
    await retriever.ingest(handbook);
    await retriever.ingest(runbook);

    const [top] = await retriever.search("how long does a rollback take");

    assert.equal(top?.document.id, "handbook");
    assert.match(top!.chunk.text, /four minutes/);
    assert.deepEqual(top!.chunk.headings, ["Deployments", "Rollbacks"]);
  });

  it("carries the source through for citation", async () => {
    const retriever = new Retriever();
    await retriever.ingest(handbook);
    const [top] = await retriever.search("rollback");
    assert.equal(top?.document.source, "docs/handbook.md");
    assert.equal(top?.chunk.id.startsWith("handbook#"), true);
  });

  it("falls back to lexical when no embedder is configured", async () => {
    const retriever = new Retriever();
    await retriever.ingest(handbook);

    const hits = await retriever.search("oncall rotation", { mode: "hybrid" });

    assert.ok(hits.length > 0);
    assert.ok(hits.every((hit) => hit.ranks.dense === null));
    assert.ok(hits.some((hit) => hit.ranks.lexical !== null));
  });

  it("refuses dense retrieval without an embedder instead of silently degrading", async () => {
    const retriever = new Retriever();
    await retriever.ingest(handbook);
    await assert.rejects(
      () => retriever.search("anything", { mode: "dense" }),
      /requires an embedder/,
    );
  });

  it("runs both retrievers when an embedder is present", async () => {
    const embedder = new BagOfWordsEmbedder([
      "rollback", "image", "tag", "oncall", "rotation", "failover", "replica",
      "backups", "snapshots", "nightly", "deployments", "releases",
    ]);
    const retriever = new Retriever({ embedder });
    await retriever.ingest(handbook);
    await retriever.ingest(runbook);

    const hits = await retriever.search("failover replica");

    assert.equal(hits[0]?.document.id, "runbook");
    assert.ok(
      hits.some((hit) => hit.ranks.dense !== null),
      "the dense list contributed",
    );
  });

  it("embeds queries as queries, not as documents", async () => {
    // Voyage scores a query embedded as a passage measurably worse, so the
    // distinction has to survive the call.
    const embedder = new BagOfWordsEmbedder(["failover", "replica"]);
    const retriever = new Retriever({ embedder });
    await retriever.ingest(runbook);
    embedder.calls.length = 0;

    await retriever.search("failover");

    assert.deepEqual(embedder.calls, [{ count: 1, kind: "query" }]);
  });

  it("respects the limit", async () => {
    const retriever = new Retriever();
    await retriever.ingest(handbook);
    await retriever.ingest(runbook);
    const hits = await retriever.search("the", { limit: 2 });
    assert.ok(hits.length <= 2);
  });

  it("returns nothing from an empty corpus", async () => {
    assert.deepEqual(await new Retriever().search("anything"), []);
  });
});

describe("formatContext", () => {
  it("labels each passage with a citable chunk id", async () => {
    const retriever = new Retriever();
    await retriever.ingest(handbook);
    const context = formatContext(await retriever.search("rollback", { limit: 1 }));

    assert.match(context, /\[handbook#\d+\]/);
    assert.match(context, /Engineering Handbook/);
    assert.match(context, /Deployments > Rollbacks/);
  });

  it("says so plainly when nothing matched", () => {
    assert.match(formatContext([]), /No matching passages/);
  });
});
