import {
  Inject,
  Injectable,
  Logger,
  ServiceUnavailableException,
} from "@nestjs/common";
import { Agent, defineTool, RefusalError } from "@dhairya-007/agent-harness";
import { formatContext, type RetrievedChunk } from "@rag/core";
import { z } from "zod";
import { CorpusService } from "./corpus.service.js";

export interface Citation {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  headings: string[];
  source?: string;
  excerpt: string;
}

export interface Answer {
  answer: string;
  citations: Citation[];
  steps: number;
  verified: boolean | null;
  usage: { inputTokens: number; outputTokens: number };
  /** Searches the agent chose to run, in order. */
  searches: string[];
}

const SYSTEM = [
  "You answer questions using only the passages returned by the search_corpus tool.",
  "",
  "Search before you answer, and search again with different wording if the first",
  "results are thin — you are not limited to one query.",
  "",
  "Cite every claim with the bracketed chunk id of the passage it came from, like",
  "[handbook#3]. If the passages do not contain the answer, say so plainly and name",
  "what is missing. Never fill a gap from your own knowledge: an unsupported answer",
  "is worse than no answer here, because the user cannot tell the difference.",
].join("\n");

/**
 * Two copies on purpose. `matchAll` requires the global flag, but `test` on a
 * global regex advances `lastIndex` and so alternates true/false across calls —
 * which here would mean the verifier passing every other answer.
 */
const CITATION_ALL = /\[([\w./-]+#\d+)\]/g;
const HAS_CITATION = /\[[\w./-]+#\d+\]/;

/**
 * Answers questions over the corpus.
 *
 * Retrieval is a *tool*, not a prefix stuffed into the prompt. A single
 * retrieve-then-generate pass is stuck with whatever the user's phrasing
 * happened to match; giving the model a search tool lets it rephrase, widen,
 * and go back for more when the first results are thin — which is most of the
 * difference between a demo and something that answers real questions.
 */
@Injectable()
export class AnswerService {
  private readonly logger = new Logger(AnswerService.name);

  // Explicit token — see the note in RagController.
  constructor(@Inject(CorpusService) private readonly corpus: CorpusService) {}

  async ask(question: string, options: { limit?: number } = {}): Promise<Answer> {
    if (!process.env["ANTHROPIC_API_KEY"] && !process.env["ANTHROPIC_AUTH_TOKEN"]) {
      throw new ServiceUnavailableException(
        "no Anthropic credentials configured; /search works without them",
      );
    }

    const retrieved = new Map<string, RetrievedChunk>();
    const searches: string[] = [];

    const searchCorpus = defineTool({
      name: "search_corpus",
      description:
        "Searches the indexed documents and returns the most relevant passages, " +
        "each labelled with a chunk id you can cite. Call it more than once with " +
        "different phrasings if the first results do not answer the question.",
      input: z.object({
        query: z
          .string()
          .min(2)
          .describe("What to look for. Natural language or keywords both work."),
        limit: z
          .number()
          .int()
          .min(1)
          .max(20)
          .optional()
          .describe("How many passages to return. Defaults to 8."),
      }),
      run: async ({ query, limit }) => {
        searches.push(query);
        const results = await this.corpus.search(query, {
          limit: limit ?? options.limit ?? 8,
        });
        // Remember everything the agent saw, so a citation can be resolved to
        // a real passage afterwards instead of taken on trust.
        for (const result of results) retrieved.set(result.chunk.id, result);
        return formatContext(results);
      },
    });

    const agent = new Agent({
      system: SYSTEM,
      tools: [searchCorpus],
      effort: "medium",
      maxSteps: 8,
      budget: { maxInputTokens: 150_000, keepRecentResults: 3 },
      // The failure this guards against is a fluent, confident answer with no
      // passage behind it. Cheap to check, and the whole point of the service.
      verify: ({ text }) =>
        HAS_CITATION.test(text) || /\b(not|no) (in|found|covered)\b/i.test(text)
          ? { ok: true }
          : {
              ok: false,
              feedback:
                "That answer cites no passage. Either cite the chunk ids you " +
                "relied on, or say the corpus does not cover this.",
            },
    });

    try {
      const result = await agent.run(question);
      return {
        answer: result.text,
        citations: resolveCitations(result.text, retrieved),
        steps: result.steps,
        verified: result.verified,
        usage: {
          inputTokens: result.usage.inputTokens,
          outputTokens: result.usage.outputTokens,
        },
        searches,
      };
    } catch (error) {
      if (error instanceof RefusalError) {
        this.logger.warn(`model declined: ${error.category ?? "unknown"}`);
        throw new ServiceUnavailableException(
          `the model declined this request${error.category ? ` (${error.category})` : ""}`,
        );
      }
      throw error;
    }
  }
}

/**
 * Turns `[chunk#3]` markers in the answer into real passages.
 *
 * Only ids the agent actually retrieved are resolved. A model can invent a
 * plausible-looking citation, and an invented one that rendered as a working
 * link would be worse than no citation at all.
 */
function resolveCitations(
  answer: string,
  retrieved: ReadonlyMap<string, RetrievedChunk>,
): Citation[] {
  const seen = new Set<string>();
  const citations: Citation[] = [];

  for (const match of answer.matchAll(CITATION_ALL)) {
    const chunkId = match[1]!;
    if (seen.has(chunkId)) continue;
    seen.add(chunkId);

    const result = retrieved.get(chunkId);
    if (!result) continue;

    citations.push({
      chunkId,
      documentId: result.document.id,
      documentTitle: result.document.title,
      headings: result.chunk.headings,
      ...(result.document.source ? { source: result.document.source } : {}),
      excerpt:
        result.chunk.text.length > 300
          ? `${result.chunk.text.slice(0, 300)}…`
          : result.chunk.text,
    });
  }

  return citations;
}
