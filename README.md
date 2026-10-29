# rag-service

Ask questions over your own documents and get answers that **cite the passages they came from** — or admit the corpus does not cover it.

A full-stack TypeScript monorepo: a retrieval engine with no dependencies, a NestJS API, and a Next.js front end. The answering layer is built on [agent-harness](https://github.com/Dhairya-007/agent-harness), so retrieval is a *tool the model can call repeatedly*, not a block of text stapled to the prompt.

```
┌─────────────┐   ┌──────────────────┐   ┌─────────────────────────┐
│  Next.js    │──▶│  NestJS API      │──▶│  @rag/core              │
│  app router │   │  /documents      │   │  chunk · BM25 · vectors │
│             │   │  /search  /ask   │   │  · RRF  · retrieve      │
└─────────────┘   └────────┬─────────┘   └─────────────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  agent-harness   │  search_corpus tool,
                  │  Claude Opus 5.5 │  citation verifier
                  └──────────────────┘
```

---

## Why it is built this way

**Retrieval is a tool, not a prompt prefix.** The usual shape is retrieve-once, stuff the results into the prompt, generate. That pipeline is stuck with whatever the user's phrasing happened to match. Here the model gets a `search_corpus` tool and can rephrase, widen the query, and go back for more when the first results are thin — which is most of the distance between a demo and something that answers real questions. `result.searches` shows what it actually asked.

**Hybrid retrieval, fused by rank.** BM25 finds the exact error code or flag someone pasted in; dense vectors find the paraphrase. Combining their *scores* is a category error — BM25 scores are unbounded sums of idf terms, cosine lives in [-1, 1] — so they are combined by **rank** instead, with reciprocal rank fusion. A passage both retrievers like beats one that only one of them loves.

**An uncited answer is a failure, not a response.** The agent runs behind a verifier that rejects any answer with no `[chunk#id]` marker and sends the model back. Citations in the response are then resolved against what was *actually retrieved* — a model can invent a plausible-looking id, and rendering an invented citation as a working source would be worse than showing none.

**Dense embeddings are optional.** The Claude API has no embeddings endpoint; Claude is the generation half of RAG. `Embedder` is an interface with a Voyage adapter. Without a key the service runs BM25-only, which is a real retrieval system and keeps the whole repo runnable, testable and free.

---

## Running it

```bash
npm install
npm run build --workspace @rag/core   # the API imports core's build output
npm test                               # 52 tests, no network, no API key

# terminal 1 — API on :3001
npm run dev:api

# terminal 2 — UI on :3000
npm run dev:web
```

Open <http://localhost:3000>, paste a markdown document, and ask it something.

**Search works with no credentials at all.** Only `/ask` calls a model.

| Variable | Needed for | Default |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | `/ask` — grounded answers | — (search still works) |
| `VOYAGE_API_KEY` | hybrid retrieval | — (BM25-only) |
| `PORT` | API port | `3001` |
| `CORS_ORIGINS` | comma-separated allowed origins | `http://localhost:3000` |
| `NEXT_PUBLIC_API_URL` | where the UI looks for the API | `http://localhost:3001` |

---

## API

| | | |
| --- | --- | --- |
| `GET` | `/health` | status plus corpus size and whether retrieval is hybrid |
| `POST` | `/documents` | `{ id, title, text, source? }` → `{ id, chunks }` |
| `GET` | `/documents` | indexed documents |
| `DELETE` | `/documents/:id` | removes a document and its chunks |
| `POST` | `/search` | `{ query, limit?, mode? }` → ranked passages. **No model call.** |
| `POST` | `/ask` | `{ question, limit? }` → answer, citations, usage, searches |

```bash
curl -X POST localhost:3001/documents -H 'content-type: application/json' -d '{
  "id": "handbook", "title": "Engineering Handbook",
  "text": "# Deployments\n\nReleases are cut on Tuesdays.\n\n## Rollbacks\n\nA rollback takes about four minutes."
}'

curl -X POST localhost:3001/search -H 'content-type: application/json' \
  -d '{"query": "how long does a rollback take"}'
```

```json
{
  "results": [{
    "chunkId": "handbook#1",
    "documentTitle": "Engineering Handbook",
    "headings": ["Deployments", "Rollbacks"],
    "text": "A rollback takes about four minutes.",
    "score": 0.0164,
    "ranks": { "lexical": 1, "dense": null }
  }]
}
```

Validation is strict: unknown fields are **rejected**, not ignored, so a typo'd parameter fails loudly instead of silently doing nothing.

---

## `@rag/core`

Zero runtime dependencies. Usable on its own.

### Chunking

Splits on **structure before length** — a heading is a real semantic boundary, character 1200 is not — and carries the heading path onto every chunk. "Rate limits" is meaningless alone; `API > Errors > Rate limits` is retrievable and gives the model something to cite.

Short sections are **not** merged by default. Folding is opt-in via `minChars`, and even then a section only folds into its own **parent** — merging "Oncall" into "Deployments" because it happened to be short would file unrelated content under the wrong heading and make the breadcrumb a lie.

### BM25

Tokenization keeps identifiers whole: `max_tokens`, `claude-opus-5-5` and `v1.2.3` survive as single terms rather than shattering into noise. There is deliberately **no stemmer** — a stemmer is a per-language asset, and a bad one (collapsing "universe" and "university") costs more precision than it buys recall.

The idf uses the `log(1 + …)` form, which stays non-negative. The textbook version goes negative for terms appearing in over half the corpus, which on a small corpus means a common word actively pushes documents *down*.

### Vectors

`InMemoryVectorIndex` is an exhaustive scan. For a product's documentation — a few thousand chunks — a scan is both fast enough and exactly correct, where an approximate index trades recall for a speedup nobody would notice. Past ~10⁵ vectors, swap it; `Retriever` only needs `search`.

---

## What is not here

**Persistence.** The corpus lives in memory and dies with the process. The storage seam is `Retriever`'s two index calls, so pgvector is a contained change — but shipping a Postgres adapter I could not run would be worse than shipping none, so it is not in the box.

**Re-ranking.** A cross-encoder pass over the fused top-k is the next real quality win.

**Auth and rate limiting.** This is a reference implementation, not a deployment.

---

## Tests

```bash
npm test
```

52 tests, no network and no API keys:

- **`@rag/core` (40)** — chunking boundaries and heading paths, BM25 ranking and the non-negativity property, cosine edge cases, RRF's scale-invariance and weighting, and the retriever's ingest/replace/remove behaviour with a deterministic offline embedder as a double.
- **`@rag/api` (12)** — the real Nest app over HTTP: ingest, list, search, delete, DTO validation rejecting unknown and out-of-range fields, and `/ask` explaining itself when no credentials are set.

API tests compile with `tsc` before running, rather than being transpiled on the fly. That is not a style preference: Nest's dependency injection and `ValidationPipe` both read `design:paramtypes` metadata, which esbuild does not emit — so a transpile-only test run silently loses validation and injects `undefined`. Compiling also means the tests exercise the artifact that actually ships.

---

## License

MIT © Dhairya Oza
