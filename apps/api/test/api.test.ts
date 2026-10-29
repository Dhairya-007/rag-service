import "reflect-metadata";
import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";
import { ValidationPipe, type INestApplication } from "@nestjs/common";
import { Test } from "@nestjs/testing";
import request from "supertest";
import { AppModule } from "../src/app.module.js";

/**
 * Exercises the HTTP surface end to end — routing, validation, retrieval —
 * against the real Nest app. Nothing here calls a model: `/search` is pure
 * retrieval, and `/ask` is only checked for the no-credentials path.
 */
describe("RAG API", () => {
  let app: INestApplication;

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
      "The oncall rotation is weekly and pages route to the primary engineer first.",
    ].join("\n"),
  };

  before(async () => {
    const moduleRef = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleRef.createNestApplication();
    app.useGlobalPipes(
      new ValidationPipe({
        whitelist: true,
        forbidNonWhitelisted: true,
        transform: true,
        transformOptions: { enableImplicitConversion: true },
      }),
    );
    await app.init();
  });

  after(async () => {
    await app?.close();
  });

  const http = () => request(app.getHttpServer());

  it("reports health and an empty corpus", async () => {
    const response = await http().get("/health").expect(200);
    assert.equal(response.body.status, "ok");
    assert.equal(response.body.corpus.documents, 0);
    assert.equal(typeof response.body.corpus.hybrid, "boolean");
  });

  it("ingests a document and reports its chunk count", async () => {
    const response = await http().post("/documents").send(handbook).expect(201);
    assert.equal(response.body.id, "handbook");
    assert.ok(response.body.chunks > 0);

    const health = await http().get("/health").expect(200);
    assert.equal(health.body.corpus.documents, 1);
  });

  it("lists documents without dumping their full text", async () => {
    const response = await http().get("/documents").expect(200);
    assert.deepEqual(response.body.documents, [
      { id: "handbook", title: "Engineering Handbook", source: "docs/handbook.md" },
    ]);
  });

  it("searches and returns citable chunks", async () => {
    const response = await http()
      .post("/search")
      .send({ query: "how long does a rollback take" })
      .expect(200);

    const [top] = response.body.results;
    assert.equal(top.documentId, "handbook");
    assert.match(top.chunkId, /^handbook#\d+$/);
    assert.match(top.text, /four minutes/);
    assert.deepEqual(top.headings, ["Deployments", "Rollbacks"]);
    assert.equal(typeof top.score, "number");
  });

  it("honours the limit", async () => {
    const response = await http()
      .post("/search")
      .send({ query: "the pipeline rotation rollback", limit: 1 })
      .expect(200);
    assert.equal(response.body.results.length, 1);
  });

  it("rejects an unknown field instead of ignoring it", async () => {
    await http()
      .post("/search")
      .send({ query: "rollback", limitt: 3 })
      .expect(400);
  });

  it("rejects a bad document id", async () => {
    await http()
      .post("/documents")
      .send({ id: "not a valid id!", title: "x", text: "y" })
      .expect(400);
  });

  it("rejects an out-of-range limit", async () => {
    await http().post("/search").send({ query: "x y", limit: 999 }).expect(400);
  });

  it("rejects dense retrieval when no embedder is configured", async () => {
    // Surfaced as a 500 rather than silently serving lexical results.
    await http()
      .post("/search")
      .send({ query: "rollback", mode: "dense" })
      .expect(500);
  });

  it("explains itself when asked without Anthropic credentials", async (t) => {
    if (process.env["ANTHROPIC_API_KEY"] || process.env["ANTHROPIC_AUTH_TOKEN"]) {
      t.skip("credentials present — this path only covers the unconfigured case");
      return;
    }
    const response = await http()
      .post("/ask")
      .send({ question: "how long does a rollback take?" })
      .expect(503);
    assert.match(response.body.message, /credentials/);
  });

  it("deletes a document and forgets its chunks", async () => {
    await http().delete("/documents/handbook").expect(204);

    const health = await http().get("/health").expect(200);
    assert.equal(health.body.corpus.documents, 0);
    assert.equal(health.body.corpus.chunks, 0);

    const search = await http().post("/search").send({ query: "rollback" }).expect(200);
    assert.deepEqual(search.body.results, []);
  });

  it("404s when deleting something that is not there", async () => {
    await http().delete("/documents/nope").expect(404);
  });
});
