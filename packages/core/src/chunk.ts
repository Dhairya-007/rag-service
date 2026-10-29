/**
 * Splitting documents into retrievable pieces.
 *
 * Chunking is the part of a RAG system that decides what retrieval can
 * possibly find. Two rules drive the implementation below:
 *
 * 1. Split on structure before length. A heading boundary is a real semantic
 *    break; character 1200 is not. Splitting mid-sentence to hit a round
 *    number produces chunks that retrieve well and read badly.
 * 2. Carry the heading path into the chunk text. "Rate limits" means nothing
 *    on its own; "API > Errors > Rate limits" is retrievable and gives the
 *    model somewhere to cite.
 */

export interface Chunk {
  id: string;
  documentId: string;
  /** Position within the document, 0-based. */
  index: number;
  /** The chunk body, without the heading breadcrumb. */
  text: string;
  /** Heading breadcrumb, e.g. `["API", "Errors"]`. */
  headings: string[];
  /** Character offsets into the original document. */
  start: number;
  end: number;
}

export interface ChunkOptions {
  /** Soft ceiling. A section under this stays whole. Default 1200. */
  maxChars?: number;
  /** Characters repeated from the previous chunk. Default 150. */
  overlapChars?: number;
  /**
   * Fold a section shorter than this into its **parent** section. Default 0,
   * meaning no folding: a two-line section under a precise heading is more
   * retrievable on its own than buried in a larger one, so merging is opt-in.
   */
  minChars?: number;
}

const DEFAULTS = { maxChars: 1200, overlapChars: 150, minChars: 0 } as const;

interface Section {
  headings: string[];
  text: string;
  start: number;
}

const HEADING = /^(#{1,6})\s+(.*)$/;

/** Splits markdown into sections, tracking the heading path to each one. */
function sections(markdown: string): Section[] {
  const lines = markdown.split("\n");
  const out: Section[] = [];
  const path: string[] = [];

  let buffer: string[] = [];
  let start = 0;
  let offset = 0;

  const flush = (): void => {
    const text = buffer.join("\n").trim();
    if (text) out.push({ headings: [...path], text, start });
    buffer = [];
  };

  for (const line of lines) {
    const match = HEADING.exec(line);
    if (match) {
      flush();
      const depth = match[1]!.length;
      const title = match[2]!.trim();
      path.length = Math.min(path.length, depth - 1);
      path[depth - 1] = title;
      for (let i = 0; i < depth - 1; i++) path[i] ??= "";
      start = offset + line.length + 1;
    } else {
      if (buffer.length === 0) start = offset;
      buffer.push(line);
    }
    offset += line.length + 1;
  }
  flush();

  return out.map((section) => ({
    ...section,
    headings: section.headings.filter((heading) => heading !== ""),
  }));
}

/** Packs paragraphs into pieces no larger than `maxChars` where it can. */
function pack(text: string, maxChars: number, overlapChars: number): string[] {
  if (text.length <= maxChars) return [text];

  const paragraphs = text.split(/\n{2,}/).filter((part) => part.trim() !== "");
  const pieces: string[] = [];
  let current = "";

  const push = (): void => {
    if (!current.trim()) return;
    pieces.push(current.trim());
    // Carry a tail forward so a fact spanning a boundary is still findable
    // from either side.
    current = overlapChars > 0 ? `${current.slice(-overlapChars)}\n\n` : "";
  };

  for (const paragraph of paragraphs) {
    // A single paragraph over the ceiling is split on sentence ends rather
    // than mid-word.
    if (paragraph.length > maxChars) {
      push();
      for (const sentence of paragraph.split(/(?<=[.!?])\s+/)) {
        if (current.length + sentence.length > maxChars) push();
        current += `${sentence} `;
      }
      continue;
    }
    if (current.length + paragraph.length > maxChars) push();
    current += `${paragraph}\n\n`;
  }
  push();

  return pieces.filter((piece) => piece !== "");
}

/** True when `parent` is the same section as, or an ancestor of, `child`. */
function isAncestorOf(parent: string[], child: string[]): boolean {
  if (parent.length > child.length) return false;
  return parent.every((heading, i) => child[i] === heading);
}

/**
 * Splits a markdown document into chunks.
 *
 * Chunk ids are `${documentId}#${index}`, which makes a citation resolvable
 * without a second lookup.
 */
export function chunkMarkdown(
  documentId: string,
  markdown: string,
  options: ChunkOptions = {},
): Chunk[] {
  const maxChars = options.maxChars ?? DEFAULTS.maxChars;
  const overlapChars = options.overlapChars ?? DEFAULTS.overlapChars;
  const minChars = options.minChars ?? DEFAULTS.minChars;

  const merged: Section[] = [];
  for (const section of sections(markdown)) {
    const previous = merged.at(-1);
    // Fold a stub only into its own parent. Merging "Oncall" into
    // "Deployments" because it happened to be short would put unrelated
    // content under the wrong heading and make the breadcrumb a lie.
    if (
      previous &&
      minChars > 0 &&
      section.text.length < minChars &&
      isAncestorOf(previous.headings, section.headings)
    ) {
      previous.text = `${previous.text}\n\n${section.text}`;
      continue;
    }
    merged.push({ ...section });
  }

  const chunks: Chunk[] = [];
  for (const section of merged) {
    let cursor = section.start;
    for (const piece of pack(section.text, maxChars, overlapChars)) {
      const index = chunks.length;
      chunks.push({
        id: `${documentId}#${index}`,
        documentId,
        index,
        text: piece,
        headings: section.headings,
        start: cursor,
        end: cursor + piece.length,
      });
      cursor += piece.length;
    }
  }

  return chunks;
}

/** What actually gets indexed: the heading path plus the body. */
export function indexableText(chunk: Chunk): string {
  return chunk.headings.length > 0
    ? `${chunk.headings.join(" > ")}\n\n${chunk.text}`
    : chunk.text;
}
