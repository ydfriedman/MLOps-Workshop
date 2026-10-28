"""A small retrieval-augmented (RAG) assistant over Ethiopian public health guidelines.

    python rag.py "What is the community case definition of suspected cholera?"
    python rag.py "..." --model qwen/qwen3.5-9b --k 8
    python rag.py "..." --retriever embed
    python rag.py "..." --docs markdown       # the Markdown versions of the PDFs (guidelines_md/)
    python rag.py "..." --docs markdown --chunking structure   # passages follow headings, not pages

How it works, in three steps:
  1. **Index** (once): read the guideline PDFs, cut each page into overlapping passages of
     about 200 words, and build a keyword index (TF-IDF) over them.
  2. **Retrieve**: for a question, find the k passages that share the most (rare) words with it, or, with
     `--retriever embed`, the k passages closest in meaning (a small embedding model).
  3. **Generate**: send the question plus those passages to an LLM, with instructions to answer
     only from them, cite them as [1], [2] ..., and say so when the answer isn't there.

Every part is a setting you can change (chunk size, k, model, prompt) and every change can make
answers better or worse: that is what evaluate.py measures.

The LLM is called through an OpenAI-compatible API. In the workshop that is OpenRouter (hosted
open-weight models, one shared key); a self-hosted server (vLLM, Ollama) works the same way:
set LLM_BASE_URL to its address.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env"                           # the API key lives here; ignored by git, so never committed
if not ENV_FILE.exists():
    ENV_FILE.write_text("# Paste the shared workshop key after the = sign, save, then restart the notebook's kernel.\n"
                        "OPENROUTER_API_KEY=\n")
try:
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
except ImportError:
    pass
REPO = HERE.parent
DOCS = REPO / "downloads" / "guidelines"          # filled by: python scripts/download.py guidelines
DOCS_MD = HERE / "guidelines_md"                  # the same PDFs as Markdown (LlamaParse, once: facilitator/parse_guidelines.py)
INDEX_CACHE = HERE / "index"                      # passages, cached after the first run

OPENROUTER = "https://openrouter.ai/api/v1"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"            # 67 MB, English; runs on onnxruntime via fastembed (no PyTorch)
REFUSAL = "The guidelines provided do not cover this."

logging.getLogger("pypdf").setLevel(logging.ERROR)   # some PDFs trigger harmless parser warnings


@dataclass
class Settings:
    """Everything that defines one version of the assistant. Logged with every evaluation run."""
    model: str = "google/gemma-4-26b-a4b-it"      # open-weight (Apache-2.0), hosted on OpenRouter
    retriever: str = "tfidf"                       # "tfidf" (shared words) or "embed" (meaning, EMBED_MODEL)
    k: int = 5                                     # passages given to the model
    chunk_words: int = 200                         # passage length
    overlap_words: int = 40                        # overlap between consecutive passages
    prompt_version: str = "v1"
    temperature: float = 0.0                       # as repeatable as the provider allows
    thinking: bool = False                         # "reasoning" models think before answering: slower, costlier
    docs: str = "pdf"                              # "pdf" (text pulled from the PDFs) or "markdown" (guidelines_md/)
    chunking: str = "fixed"                        # "fixed" (word windows, one page at a time) or "structure" (markdown)


CHUNKINGS = ["fixed", "structure"]                 # "semantic" (split inside sections by meaning) would go here


def run_params(settings):
    """The settings as saved with a run. `docs` and `chunking` were added later: runs on the PDFs keep their key."""
    params = asdict(settings)
    if params["docs"] == "pdf":
        del params["docs"], params["chunking"]
    return params


PROMPTS = {
    "v1": (
        "You answer questions from health workers in Ethiopia using ONLY the numbered guideline "
        "passages provided.\n"
        "- Base every statement on the passages and cite them like [1] or [2][3].\n"
        "- If the passages do not contain the answer, reply exactly: \"" + REFUSAL + "\"\n"
        "- Be concise: at most 4 sentences. Quote numbers, doses and cut-offs exactly.\n"
        "- Answer in the language of the question."
    ),
}


# ---------------------------------------------------------------- 1. index


def document_titles():
    """File stem -> human-readable title, from the download manifest."""
    manifest = json.loads((REPO / "scripts" / "downloads.json").read_text())
    return {Path(i["file"]).stem: i["description"] for i in manifest["guidelines"]}


def load_pages(folder=DOCS):
    """One row per PDF page: doc (file stem), page (1-based), text."""
    from pypdf import PdfReader

    files = sorted(Path(folder).glob("*.pdf"))
    if not files:
        raise FileNotFoundError(f"no PDFs in {folder}: run `python scripts/download.py guidelines` from the repo root")
    rows = []
    for f in files:
        for i, page in enumerate(PdfReader(f).pages, 1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if len(text.split()) >= 20:          # skip blank and image-only pages
                rows.append({"doc": f.stem, "page": i, "text": text})
    return pd.DataFrame(rows)


def make_passages(pages, chunk_words=200, overlap_words=40):
    """Cut each page into overlapping passages. A passage never crosses pages, so citations stay exact."""
    rows, step = [], chunk_words - overlap_words
    for p in pages.itertuples():
        words = p.text.split()
        for start in range(0, max(len(words) - overlap_words, 1), step):
            rows.append({"doc": p.doc, "page": p.page, "end_page": p.page,
                         "text": " ".join(words[start:start + chunk_words])})
    passages = pd.DataFrame(rows)
    passages.insert(0, "passage_id", [f"{d}:p{pg}:{i}" for i, (d, pg) in enumerate(zip(passages.doc, passages.page))])
    return passages


PAGE_MARK = re.compile(r"<!-- page (\d+)[^>]*-->")
HEADING = re.compile(r"#{1,6} ")


def markdown_blocks(text):
    """Split a Markdown file into blocks: (page, kind, text). A table or a fenced block (a flowchart) is one block."""
    blocks, page, lines = [], None, text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        mark = PAGE_MARK.match(line.strip())
        if mark:
            page, i = int(mark.group(1)), i + 1
        elif page is None or not line.strip():
            i += 1
        elif line.startswith("```"):
            j = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("```")), len(lines) - 1)
            blocks.append((page, "figure", "\n".join(lines[i:j + 1])))
            i = j + 1
        elif line.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            blocks.append((page, "table", "\n".join(lines[i:j])))
            i = j
        elif HEADING.match(line):
            blocks.append((page, "heading", line.strip()))
            i += 1
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not PAGE_MARK.match(lines[j].strip()) \
                    and not HEADING.match(lines[j]) and not lines[j].lstrip().startswith(("|", "```")):
                j += 1
            blocks.append((page, "text", "\n".join(lines[i:j])))
            i = j
    return [(p, k, re.sub(r"<br\s*/?>", " ", t)) for p, k, t in blocks]


MERMAID_NODE = re.compile(r'\b([A-Za-z]\w*)\s*[\[({]+(?:"(.*?)"|([^"\])}]*))[\])}]+')
MERMAID_EDGE = re.compile(r"([A-Za-z]\w*)\s*(?:-->|---|-\.->|==>)\s*(?:\|\s*\"?(.*?)\"?\s*\|)?\s*([A-Za-z]\w*)")


def flowchart_text(block):
    """A Mermaid flowchart (how LlamaParse writes most diagrams) as plain lines: each box, then each arrow.
    Other fenced blocks pass through unchanged."""
    if not block.startswith("```mermaid"):
        return block
    labels, lines = {}, []
    for node, quoted, bare in MERMAID_NODE.findall(block):
        label = re.sub(r"\*\*|<br\s*/?>", " ", quoted or bare).strip()
        if node not in labels and label:
            labels[node] = label
            lines.append(f"- {label}")
    def short(n):
        return re.split(r"\s\*\s|[.:]\s", labels.get(n, n))[0][:60].strip()
    for a, cond, b in MERMAID_EDGE.findall(block.replace("|>", "|")):
        lines.append(f"- {short(a)} → {short(b)}" + (f" (if {cond.strip()})" if cond.strip() else ""))
    return "\n".join(lines)


def markdown_pages(folder=DOCS_MD):
    """One row per page of the Markdown files, like load_pages() for the PDFs: for fixed-window chunking."""
    rows = []
    for f in sorted(Path(folder).glob("*.md")):
        by_page = {}
        for p, kind, text in markdown_blocks(f.read_text()):
            by_page.setdefault(p, []).append(flowchart_text(text) if kind == "figure" else text)
        for p, texts in by_page.items():
            text = re.sub(r"\s+", " ", " ".join(texts)).strip()
            if len(text.split()) >= 20:
                rows.append({"doc": f.stem, "page": p, "text": text})
    if not rows:
        raise FileNotFoundError(f"no Markdown files in {folder}")
    return pd.DataFrame(rows)


def table_cells(row):
    row = row.strip()
    return [c.strip() for c in row[row.startswith("|"):len(row) - row.endswith("|")].split("|")]


def continues(prev, table):
    """Is `table`, at the top of a page, the rest of `prev`, which ended the page before? Then the rows to append.

    Only if it has as many columns, and either repeats prev's header (dropped) or has no header of its own:
    LlamaParse then promotes the first data row to header, recognisable as a row that starts lower-case or with
    a bullet, has a number, an empty or a long (9+ words) cell, or carries on a list of abbreviations. A row that
    names a table ("Reference Table for ...") is a new table. A header reworded on the new page is not recognised:
    those tables stay apart, which is safe, only less tidy."""
    lines, prev_lines = table.splitlines(), prev.splitlines()
    head, prev_head = table_cells(lines[0]), table_cells(prev_lines[0])
    if len(head) != len(prev_head):
        return None
    if [c.lower() for c in head] == [c.lower() for c in prev_head]:
        return lines[2:]
    acronym = re.compile(r"[A-Z][A-Z0-9-]{1,9}")
    if any(re.search(r"\btable\b", c, re.I) for c in head):
        return None
    data_like = (re.match(r"[a-z]|\\?\*|•", head[0] or " ")
                 or any(not c or re.match(r"[<>≤≥]?\s*\d", c) or len(c.split()) > 8 for c in head)
                 or (acronym.fullmatch(head[0]) and acronym.fullmatch(table_cells(prev_lines[-1])[0])))
    return [lines[0]] + lines[2:] if data_like else None


FOOTNOTE = re.compile(r"(\\?\*|<sup>|\d+\s*<sup>)")


def merge_continued_tables(blocks):
    """Blocks as dicts with a page range; a table carried over a page break becomes one table (footnotes at the
    bottom of the first page, which sit between the two parts, move below it)."""
    out = []
    for p, kind, text in blocks:
        if kind == "table" and out:
            first_on_page = out[-1]["end"] < p
            back = len(out) - 1
            while back >= 0 and out[back]["kind"] == "text" and FOOTNOTE.match(out[back]["text"]) \
                    and out[back]["end"] == p - 1 and len(out) - back <= 3:
                back -= 1
            prev = out[back] if back >= 0 else None
            rows = continues(prev["text"], text) if first_on_page and prev and prev["kind"] == "table" \
                and prev["end"] == p - 1 else None
            if rows is not None:
                prev["text"] += "\n" + "\n".join(rows)
                prev["end"] = p
                continue
        out.append({"start": p, "end": p, "kind": kind, "text": text})
    return out


def heading_title(line):
    return re.sub(r"\*\*|<[^>]+>", "", line.lstrip("#")).strip()


ROMAN = re.compile(r"[IVX]+[\s.:)]+[A-Z]")         # "IV Strategies" (case matters: "ii) Case definitions" is a list)
NUMBER = re.compile(r"(\d+(?:\.\d+)*)\.?(?:\s|$)")


def heading_levels(headings):
    """Nesting depth of each heading line of one document.

    LlamaParse's #-levels are unreliable (an annex under another annex, the front matter's "Contents" above
    chapter 4); the printed numbering is not: "7.4" is 2, "7.4.1" is 3, "Annex 4" and "Chapter VI" are 1. A single
    number ("7.") is a chapter if the next dotted number after it is "7.x" or its title is in capitals; otherwise it is a numbered list
    item set as a heading ("11 Perinatal death", "2. Multibacillary Leprosy"). Where chapters are roman (I, II, ...)
    those are level 1. An unnumbered heading sits below the numbered section it is in, ordered by its #-level."""
    titles = [heading_title(h) for h in headings]
    numbers = [NUMBER.match(t).group(1) if NUMBER.match(t) else None for t in titles]
    roman = sum(bool(ROMAN.match(t)) for t in titles) >= 3
    levels = []
    for i, (line, title, number) in enumerate(zip(headings, titles, numbers)):
        if re.match(r"(annex|appendix|chapter|part)\s+[\divxlc]+\b", title, re.I) or (roman and ROMAN.match(title)):
            levels.append(1)
        elif number and "." in number:
            levels.append(number.count(".") + 1)
        elif number and (next((n for n in numbers[i + 1:] if n and "." in n), "").split(".")[0] == number
                         or title.isupper()):
            levels.append(1)
        else:
            levels.append(10 + len(line) - len(line.lstrip("#")))
    return levels


def make_structure_passages(folder=DOCS_MD, chunk_words=200, overlap_words=40):
    """Passages that follow the document's structure, not its pages.

    A section (a heading and what follows, up to the next heading) is one passage if it fits in chunk_words;
    a longer one is cut at paragraph breaks (then line breaks; word windows only for a single huge line); small
    sections are packed together with the next. A table or figure is never cut, and a table continued on the next
    page is first joined into one. Each passage starts with its heading path ("4 Treatment > 4.2 Plan C") and
    keeps the range of PDF pages it covers (page .. end_page) for citations.
    """
    files = sorted(Path(folder).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no Markdown files in {folder}")
    rows, step = [], chunk_words - overlap_words
    for f in files:
        blocks = merge_continued_tables(markdown_blocks(f.read_text()))
        levels = iter(heading_levels([b["text"] for b in blocks if b["kind"] == "heading"]))
        doc_rows, path, current, passage_path = [], [], [], ""

        def words():
            return sum(len(b["text"].split()) for b in current)

        def flush():
            if current and words() >= 5:
                body = "\n\n".join(b["text"] for b in current)
                doc_rows.append({"doc": f.stem, "page": min(b["start"] for b in current),
                                 "end_page": max(b["end"] for b in current),
                                 "text": f"{passage_path}\n\n{body}" if passage_path else body, "n": words()})
            current.clear()

        def add(block, can_split_before=True):
            nonlocal passage_path
            n = len(block["text"].split())
            if current and words() + n > chunk_words and (block["kind"] == "text" or words() >= chunk_words // 2) \
                    and can_split_before:
                flush()
            if not current:
                passage_path = " > ".join(t for _, t in path if t)
            current.append(block)

        for b in blocks:
            if b["kind"] == "heading":
                level = next(levels)
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, heading_title(b["text"])))
                if words() >= chunk_words // 2:
                    flush()
                if current:                          # a small section packed with this one: keep its heading
                    current.append(b)
                continue
            if b["kind"] == "figure":
                b = dict(b, text=flowchart_text(b["text"]))
            if b["kind"] == "text" and len(b["text"].split()) > chunk_words:     # a long paragraph or list
                for line in b["text"].splitlines():
                    w = line.split()
                    if len(w) > chunk_words:
                        for start in range(0, max(len(w) - overlap_words, 1), step):
                            flush()
                            add(dict(b, text=" ".join(w[start:start + chunk_words])))
                    elif w:
                        add(dict(b, text=line))
                continue
            add(b)
        flush()
        for r in doc_rows:                            # a tiny leftover joins the passage before it
            if r["n"] < 30 and rows and rows[-1]["doc"] == r["doc"]:
                prev = rows[-1]
                prev["text"] += "\n\n" + r["text"].split("\n\n", 1)[-1]
                prev["end_page"] = max(prev["end_page"], r["end_page"])
                prev["page"] = min(prev["page"], r["page"])
                prev["n"] += r["n"]
            else:
                rows.append(r)
    passages = pd.DataFrame(rows).drop(columns="n")
    passages.insert(0, "passage_id", [f"{d}:p{pg}:{i}" for i, (d, pg) in enumerate(zip(passages.doc, passages.page))])
    return passages


def page_label(page, end_page):
    return f"page {page}" if page == end_page else f"pages {page}–{end_page}"


def source_ids(hits):
    """doc:p28 or doc:p28-29, one per passage: what a run saves as its sources."""
    return [f"{d}:p{p}" + (f"-{e}" if e != p else "") for d, p, e in zip(hits.doc, hits.page, hits.end_page)]


def on_pages(hits, doc, pages):
    """For each passage: is it from `doc`, and does its page range include one of `pages`?"""
    return [d == doc and any(p <= x <= e for x in pages) for d, p, e in zip(hits.doc, hits.page, hits.end_page)]


def fingerprint(passages):
    """Short hash of the indexed text: changes whenever a document, or the chunking, changes."""
    return hashlib.md5(pd.util.hash_pandas_object(passages[["doc", "page", "text"]], index=False).values.tobytes()).hexdigest()[:12]


class Retriever:
    """Keyword search (TF-IDF with word pairs). Simple, fast, no extra model to download.

    A dense "embedding" retriever (below) finds paraphrases better, and is weaker on exact words: drug names, doses.
    """

    def __init__(self, passages):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.passages = passages.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.passages.text)

    def search(self, question, k=5):
        scores = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return self.passages.iloc[top].assign(score=scores[top])


class EmbeddingRetriever:
    """Dense search: passages and question become vectors from a small embedding model; the nearest vectors win.

    Finds passages that say the same thing in other words. The model downloads on first use; embedding all
    passages takes a few minutes on a laptop CPU, once: the vectors are cached in index/.
    """

    def __init__(self, passages, model=EMBED_MODEL):
        from fastembed import TextEmbedding

        self.passages = passages.reset_index(drop=True)
        self.model = TextEmbedding(model, cache_dir=str(INDEX_CACHE / "models"))
        cache = INDEX_CACHE / f"vectors_{model.split('/')[-1]}_{fingerprint(self.passages)}.npy"
        if cache.exists():
            self.matrix = np.load(cache)
        else:
            print(f"embedding {len(self.passages)} passages with {model} (once; a few minutes)...", flush=True)
            m = np.array(list(self.model.passage_embed(self.passages.text.tolist())))
            self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
            np.save(cache, self.matrix)

    def search(self, question, k=5):
        q = np.array(next(iter(self.model.query_embed([question]))))
        scores = self.matrix @ (q / np.linalg.norm(q))
        top = np.argsort(-scores)[:k]
        return self.passages.iloc[top].assign(score=scores[top])


def build(settings=Settings(), folder=DOCS):
    """Load (or reuse) the passages for these chunk settings and return a Retriever."""
    INDEX_CACHE.mkdir(exist_ok=True)
    if settings.chunking not in CHUNKINGS or (settings.docs == "pdf" and settings.chunking != "fixed"):
        raise ValueError(f"chunking {settings.chunking!r}: choose from {CHUNKINGS}; 'structure' needs docs='markdown'")
    cache = INDEX_CACHE / f"passages_{settings.chunk_words}_{settings.overlap_words}.parquet"
    # the Markdown passages are a second's work: rebuilt every time, so an edited file is never stale
    if settings.docs == "markdown" and settings.chunking == "structure":
        passages = make_structure_passages(DOCS_MD, settings.chunk_words, settings.overlap_words)
    elif settings.docs == "markdown":
        passages = make_passages(markdown_pages(DOCS_MD), settings.chunk_words, settings.overlap_words)
    elif cache.exists():
        passages = pd.read_parquet(cache)
        if "end_page" not in passages:              # cached before passages had a page range
            passages["end_page"] = passages.page
    else:
        passages = make_passages(load_pages(folder), settings.chunk_words, settings.overlap_words)
        passages.to_parquet(cache, index=False)
    return EmbeddingRetriever(passages) if settings.retriever == "embed" else Retriever(passages)


# ---------------------------------------------------------------- 2-3. retrieve and generate


def llm_client(async_=False):
    """OpenAI-compatible client for OpenRouter (or LLM_BASE_URL, e.g. a self-hosted vLLM server)."""
    from openai import AsyncOpenAI, OpenAI

    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError(f"no API key: open {ENV_FILE}, paste the shared key after OPENROUTER_API_KEY=, save, "
                           "and restart (the notebook's kernel, or the script)")
    base_url = os.environ.get("LLM_BASE_URL", OPENROUTER)
    return (AsyncOpenAI if async_ else OpenAI)(base_url=base_url, api_key=key, timeout=120, max_retries=3)


def thinking_switch(on=False):
    """Extra request arguments that turn a reasoning model's "thinking" on or off.

    Providers spell it differently: OpenRouter takes a `reasoning` object; Ollama and vLLM take
    OpenAI's `reasoning_effort`. Checked on OpenRouter: gemma-4-26b and qwen3.5-9b accept it (0 reasoning
    tokens when off); gpt-oss-120b refuses `enabled: False` (400), so the judge never sends it there.
    """
    if os.environ.get("LLM_BASE_URL", OPENROUTER).startswith(OPENROUTER):
        return {"extra_body": {"reasoning": {"enabled": on}}}
    return {} if on else {"reasoning_effort": "none"}


def format_passages(hits):
    return "\n\n".join(f"[{i}] ({h.doc}, {page_label(h.page, h.end_page)}) {h.text}"
                       for i, h in enumerate(hits.itertuples(), 1))


def answer(question, retriever, settings=Settings(), client=None):
    """Retrieve, then generate. Returns the answer plus everything needed to evaluate it."""
    hits = retriever.search(question, settings.k)
    client = client or llm_client()
    t0 = time.time()
    response = client.chat.completions.create(
        model=settings.model,
        temperature=settings.temperature,
        **thinking_switch(settings.thinking),
        messages=[{"role": "system", "content": PROMPTS[settings.prompt_version]},
                  {"role": "user", "content": f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question}"}],
    )
    return {
        "question": question,
        "answer": (response.choices[0].message.content or "").strip(),
        "contexts": hits.text.tolist(),
        "sources": source_ids(hits),
        "seconds": round(time.time() - t0, 2),
        "tokens": getattr(response.usage, "total_tokens", None),
        "settings": run_params(settings),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--model", default=Settings.model)
    ap.add_argument("--k", type=int, default=Settings.k)
    ap.add_argument("--retriever", default=Settings.retriever, choices=["tfidf", "embed"])
    ap.add_argument("--docs", default=Settings.docs, choices=["pdf", "markdown"])
    ap.add_argument("--chunking", default=Settings.chunking, choices=CHUNKINGS, help="'structure' needs --docs markdown")
    ap.add_argument("--search-only", action="store_true", help="show the retrieved passages; no LLM call")
    a = ap.parse_args()
    s = Settings(model=a.model, k=a.k, retriever=a.retriever, docs=a.docs, chunking=a.chunking)
    retriever = build(s)
    titles = document_titles()
    hits = retriever.search(a.question, s.k)
    for i, h in enumerate(hits.itertuples(), 1):
        print(f"[{i}] {titles.get(h.doc, h.doc)}, {page_label(h.page, h.end_page)} (score {h.score:.3f})\n    {h.text[:220]}...\n")
    if not a.search_only:
        print(answer(a.question, retriever, s)["answer"])


if __name__ == "__main__":
    main()
