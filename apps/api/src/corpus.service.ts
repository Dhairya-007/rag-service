import { Injectable, Logger, NotFoundException } from "@nestjs/common";
import {
  Retriever,
  VoyageEmbedder,
  type Document,
  type Embedder,
  type RetrievalMode,
  type RetrievedChunk,
} from "@rag/core";

/**
 * Owns the corpus for the process.
 *
 * The retriever holds its index in memory, so this is a singleton and the
 * corpus lives as long as the process does. That is a deliberate limit, not an
 * oversight — see the README. Everything that would change when a real store
 * arrives is behind `Retriever`.
 */
@Injectable()
export class CorpusService {
  private readonly logger = new Logger(CorpusService.name);
  private readonly retriever: Retriever;

  constructor() {
    this.retriever = new Retriever({ ...buildEmbedder(this.logger) });
    this.logger.log(
      this.retriever.hasEmbedder
        ? "retrieval: hybrid (BM25 + dense)"
        : "retrieval: lexical only — set VOYAGE_API_KEY for hybrid",
    );
  }

  get hybrid(): boolean {
    return this.retriever.hasEmbedder;
  }

  get stats(): { documents: number; chunks: number; hybrid: boolean } {
    return {
      documents: this.retriever.documentCount,
      chunks: this.retriever.chunkCount,
      hybrid: this.retriever.hasEmbedder,
    };
  }

  async ingest(document: Document): Promise<{ id: string; chunks: number }> {
    const chunks = await this.retriever.ingest(document);
    this.logger.log(`ingested "${document.id}" as ${chunks.length} chunks`);
    return { id: document.id, chunks: chunks.length };
  }

  list(): Document[] {
    return this.retriever.documents();
  }

  remove(id: string): void {
    const existed = this.retriever.documents().some((doc) => doc.id === id);
    if (!existed) throw new NotFoundException(`no document with id "${id}"`);
    this.retriever.remove(id);
  }

  search(
    query: string,
    options: { limit?: number; mode?: RetrievalMode } = {},
  ): Promise<RetrievedChunk[]> {
    return this.retriever.search(query, options);
  }
}

/**
 * Dense retrieval is opt-in. A missing key means lexical-only, which is a
 * working configuration; a *broken* key is a misconfiguration and is logged
 * loudly rather than silently halving retrieval quality.
 */
function buildEmbedder(logger: Logger): { embedder?: Embedder } {
  if (!process.env["VOYAGE_API_KEY"]) return {};
  try {
    return { embedder: new VoyageEmbedder() };
  } catch (error) {
    logger.error(
      `VOYAGE_API_KEY is set but the embedder could not be built: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    return {};
  }
}
