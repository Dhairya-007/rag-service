import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  Inject,
  Param,
  Post,
} from "@nestjs/common";
import { AnswerService, type Answer } from "./answer.service.js";
import { CorpusService } from "./corpus.service.js";
import { AskDto, IngestDto, SearchDto } from "./dto.js";

@Controller()
export class RagController {
  // Injected by explicit token rather than by parameter type. Type-based
  // injection reads `design:paramtypes`, which only tsc emits — esbuild (and
  // so tsx, and anything built on it) does not, and the failure mode is a
  // silently undefined dependency at request time rather than a build error.
  constructor(
    @Inject(CorpusService) private readonly corpus: CorpusService,
    @Inject(AnswerService) private readonly answers: AnswerService,
  ) {}

  @Get("health")
  health(): {
    status: "ok";
    corpus: { documents: number; chunks: number; hybrid: boolean };
  } {
    return { status: "ok", corpus: this.corpus.stats };
  }

  @Post("documents")
  ingest(@Body() body: IngestDto): Promise<{ id: string; chunks: number }> {
    return this.corpus.ingest({
      id: body.id,
      title: body.title,
      text: body.text,
      ...(body.source ? { source: body.source } : {}),
    });
  }

  @Get("documents")
  list(): { documents: Array<{ id: string; title: string; source?: string }> } {
    return {
      documents: this.corpus.list().map(({ id, title, source }) => ({
        id,
        title,
        ...(source ? { source } : {}),
      })),
    };
  }

  @Delete("documents/:id")
  @HttpCode(204)
  remove(@Param("id") id: string): void {
    this.corpus.remove(id);
  }

  /** Retrieval only — no model call, so this works with no API key. */
  @Post("search")
  @HttpCode(200)
  async search(@Body() body: SearchDto) {
    const results = await this.corpus.search(body.query, {
      ...(body.limit !== undefined ? { limit: body.limit } : {}),
      ...(body.mode ? { mode: body.mode } : {}),
    });

    return {
      results: results.map((result) => ({
        chunkId: result.chunk.id,
        documentId: result.document.id,
        documentTitle: result.document.title,
        headings: result.chunk.headings,
        text: result.chunk.text,
        score: result.score,
        ranks: result.ranks,
      })),
    };
  }

  /** Retrieval plus a grounded, cited answer. Needs Anthropic credentials. */
  @Post("ask")
  @HttpCode(200)
  ask(@Body() body: AskDto): Promise<Answer> {
    return this.answers.ask(body.question, {
      ...(body.limit !== undefined ? { limit: body.limit } : {}),
    });
  }
}
