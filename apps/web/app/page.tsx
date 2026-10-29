"use client";

import { Fragment, useCallback, useEffect, useState } from "react";
import {
  api,
  ApiError,
  type AskResult,
  type DocumentSummary,
  type Health,
  type SearchHit,
} from "../lib/api";

const SAMPLE = `# Deployments

Every service ships through the pipeline. Releases are cut on Tuesdays.

## Rollbacks

A rollback reverts to the previous image tag and takes about four minutes.

# Oncall

The oncall rotation is weekly. Pages route to the primary engineer first,
and escalate to the secondary after ten minutes with no acknowledgement.`;

type Mode = "ask" | "search";

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);

  const [docId, setDocId] = useState("handbook");
  const [docTitle, setDocTitle] = useState("Engineering Handbook");
  const [docText, setDocText] = useState(SAMPLE);
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const [mode, setMode] = useState<Mode>("ask");
  const [question, setQuestion] = useState("How long does a rollback take?");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<AskResult | null>(null);
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [h, d] = await Promise.all([api.health(), api.listDocuments()]);
      setHealth(h);
      setDocuments(d.documents);
    } catch (error) {
      setIngestError(error instanceof ApiError ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function ingest() {
    setIngesting(true);
    setIngestError(null);
    try {
      await api.ingest({ id: docId.trim(), title: docTitle.trim(), text: docText });
      await refresh();
    } catch (error) {
      setIngestError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setIngesting(false);
    }
  }

  async function removeDocument(id: string) {
    try {
      await api.remove(id);
      await refresh();
    } catch (error) {
      setIngestError(error instanceof ApiError ? error.message : String(error));
    }
  }

  async function run() {
    setBusy(true);
    setQueryError(null);
    setAnswer(null);
    setHits(null);
    try {
      if (mode === "ask") setAnswer(await api.ask(question));
      else setHits((await api.search(question)).results);
    } catch (error) {
      setQueryError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  const corpusEmpty = (health?.corpus.documents ?? 0) === 0;

  return (
    <main className="page">
      <header>
        <h1>RAG Service</h1>
        <p>Ask questions over your own documents, and get answers that cite them.</p>
      </header>

      <div className="badges">
        <span className="badge">
          {health ? `${health.corpus.documents} documents` : "connecting…"}
        </span>
        <span className="badge">{health ? `${health.corpus.chunks} chunks` : "—"}</span>
        <span className={`badge ${health?.corpus.hybrid ? "on" : "off"}`}>
          {health?.corpus.hybrid
            ? "hybrid retrieval (BM25 + dense)"
            : "lexical retrieval (BM25)"}
        </span>
      </div>

      <div className="grid">
        <section className="panel">
          <h2>Corpus</h2>

          <label htmlFor="doc-id">Document id</label>
          <input
            id="doc-id"
            value={docId}
            onChange={(event) => setDocId(event.target.value)}
            placeholder="handbook"
          />

          <label htmlFor="doc-title">Title</label>
          <input
            id="doc-title"
            value={docTitle}
            onChange={(event) => setDocTitle(event.target.value)}
            placeholder="Engineering Handbook"
          />

          <label htmlFor="doc-text">Markdown</label>
          <textarea
            id="doc-text"
            value={docText}
            onChange={(event) => setDocText(event.target.value)}
          />

          <button onClick={ingest} disabled={ingesting || !docId || !docText}>
            {ingesting ? "Indexing…" : "Add to corpus"}
          </button>

          {ingestError ? <p className="error">{ingestError}</p> : null}

          <h2 style={{ marginTop: 24 }}>Indexed</h2>
          {documents.length === 0 ? (
            <p className="empty">Nothing indexed yet.</p>
          ) : (
            <ul className="doc-list">
              {documents.map((document) => (
                <li key={document.id}>
                  <span>
                    {document.title} <code>{document.id}</code>
                  </span>
                  <button className="link" onClick={() => removeDocument(document.id)}>
                    remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="panel">
          <h2>Query</h2>

          <label htmlFor="question">Question</label>
          <input
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !busy) void run();
            }}
          />

          <div className="row">
            <button
              className={mode === "ask" ? "" : "secondary"}
              onClick={() => setMode("ask")}
            >
              Ask
            </button>
            <button
              className={mode === "search" ? "" : "secondary"}
              onClick={() => setMode("search")}
            >
              Search only
            </button>
            <button onClick={run} disabled={busy || corpusEmpty || question.length < 2}>
              {busy ? "Working…" : mode === "ask" ? "Answer" : "Retrieve"}
            </button>
          </div>

          {corpusEmpty ? (
            <p className="empty" style={{ marginTop: 12 }}>
              Add a document first.
            </p>
          ) : null}

          {/* "Search only" is pure retrieval and needs no model, which makes it
              the right thing to try when /ask reports missing credentials. */}
          {queryError ? <p className="error">{queryError}</p> : null}
          {busy ? (
            <p className="spinner">
              {mode === "ask" ? "Searching and composing an answer…" : "Retrieving…"}
            </p>
          ) : null}

          {answer ? <AnswerView result={answer} /> : null}

          {hits
            ? hits.map((hit) => (
                <article className="hit" key={hit.chunkId}>
                  <div className="hit-head">
                    <span className="hit-title">
                      {hit.documentTitle}
                      {hit.headings.length > 0 ? (
                        <span className="hit-crumb"> — {hit.headings.join(" › ")}</span>
                      ) : null}
                    </span>
                    <code className="cite">{hit.chunkId}</code>
                  </div>
                  <p className="hit-text">{hit.text}</p>
                  <div className="meta">
                    <span>score {hit.score.toFixed(4)}</span>
                    <span>lexical rank {hit.ranks.lexical ?? "—"}</span>
                    <span>dense rank {hit.ranks.dense ?? "—"}</span>
                  </div>
                </article>
              ))
            : null}

          {hits?.length === 0 ? <p className="empty">Nothing matched.</p> : null}
        </section>
      </div>
    </main>
  );
}

function AnswerView({ result }: { result: AskResult }) {
  return (
    <>
      <p className="answer">{highlightCitations(result.answer)}</p>

      {result.citations.length > 0 ? (
        <>
          <h2 style={{ marginTop: 22 }}>Sources</h2>
          {result.citations.map((citation) => (
            <article className="hit" key={citation.chunkId}>
              <div className="hit-head">
                <span className="hit-title">
                  {citation.documentTitle}
                  {citation.headings.length > 0 ? (
                    <span className="hit-crumb">
                      {" "}
                      — {citation.headings.join(" › ")}
                    </span>
                  ) : null}
                </span>
                <code className="cite">{citation.chunkId}</code>
              </div>
              <p className="hit-text">{citation.excerpt}</p>
            </article>
          ))}
        </>
      ) : null}

      <div className="meta">
        <span>{result.steps} turns</span>
        <span>
          {result.searches.length} search
          {result.searches.length === 1 ? "" : "es"}
        </span>
        <span>
          {result.usage.inputTokens} in / {result.usage.outputTokens} out
        </span>
        <span>
          {result.verified === null
            ? "unverified"
            : result.verified
              ? "cites its sources"
              : "no citation found"}
        </span>
      </div>
    </>
  );
}

/**
 * Renders `[handbook#3]` markers as chips.
 *
 * Only the markers are styled; the surrounding text is inserted as text nodes,
 * never as HTML. Model output is untrusted input like any other, and the whole
 * point of this page is that it displays text a model wrote.
 */
function highlightCitations(answer: string) {
  const parts = answer.split(/(\[[\w./-]+#\d+\])/g);
  return parts.map((part, index) =>
    /^\[[\w./-]+#\d+\]$/.test(part) ? (
      <code className="cite" key={index}>
        {part.slice(1, -1)}
      </code>
    ) : (
      <Fragment key={index}>{part}</Fragment>
    ),
  );
}
