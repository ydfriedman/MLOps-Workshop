"""A chat page for the guideline assistant: ask your own questions and see exactly what came back.

    python chat_app.py                 # then open http://127.0.0.1:8000

For each question the page shows the answer, the passages the search returned (document, pages, score, full
text), and the exact prompt sent to the model. Choose the setup (text, chunking, search, number of passages,
model) on the left; tick "compare" to ask the same question with two setups side by side.
Needs OPENROUTER_API_KEY for answers; "search only" works without it.
"""
import html
import os
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import rag

app = FastAPI()
TITLES = rag.document_titles()
_retrievers = {}


class Setup(BaseModel):
    docs: str = "pdf"
    chunking: str = "fixed"
    retriever: str = "tfidf"
    k: int = 5
    model: str = rag.Settings.model


class Ask(BaseModel):
    question: str
    setups: list[Setup]
    search_only: bool = False


def retriever_for(s: Setup):
    key = (s.docs, s.chunking, s.retriever)
    if key not in _retrievers:
        _retrievers[key] = rag.build(rag.Settings(docs=s.docs, chunking=s.chunking, retriever=s.retriever))
    return _retrievers[key]


def run(question, s: Setup, search_only):
    settings = rag.Settings(docs=s.docs, chunking=s.chunking, retriever=s.retriever, k=s.k, model=s.model)
    t0 = time.time()
    hits = retriever_for(s).search(question, s.k)
    passages = [{"n": i, "doc": TITLES.get(h.doc, h.doc), "pages": rag.page_label(h.page, h.end_page),
                 "score": round(float(h.score), 3), "text": h.text} for i, h in enumerate(hits.itertuples(), 1)]
    prompt = f"SYSTEM:\n{rag.PROMPTS[settings.prompt_version]}\n\nUSER:\nPassages:\n\n{rag.format_passages(hits)}\n\nQuestion: {question}"
    out = {"setup": s.model_dump(), "passages": passages, "prompt": prompt, "answer": None}
    if not search_only:
        result = rag.answer(question, retriever_for(s), settings)
        out |= {"answer": result["answer"], "tokens": result["tokens"]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


@app.post("/ask")
def ask(req: Ask):
    try:
        return {"results": [run(req.question, s, req.search_only) for s in req.setups]}
    except Exception as e:  # noqa: BLE001 - show the problem on the page instead of a blank error
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE.replace("{{MODEL}}", html.escape(rag.Settings.model))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guideline assistant</title>
<style>
 :root { --ink:#1b2430; --soft:#4a5561; --line:#ddd8cc; --bg:#f6f4ef; --card:#fffdf8; --teal:#0f6e68; --teal-bg:#e3f1ef; --warn:#b4501a; }
 * { box-sizing:border-box; }
 body { margin:0; font:16px/1.5 system-ui, -apple-system, "Segoe UI", Arial, sans-serif; color:var(--ink); background:var(--bg); }
 header { padding:14px 20px; border-bottom:1px solid var(--line); background:var(--card); }
 header h1 { margin:0; font-size:20px; } header p { margin:2px 0 0; color:var(--soft); font-size:14px; }
 main { display:grid; grid-template-columns:300px 1fr; gap:20px; padding:20px; }
 @media (max-width: 900px) { main { grid-template-columns:1fr; } }
 aside, .result { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }
 label { display:block; font-size:13px; color:var(--soft); margin:10px 0 3px; }
 select, input[type=number], textarea { width:100%; padding:7px; border:1px solid var(--line); border-radius:6px; font:inherit; background:#fff; }
 textarea { min-height:90px; resize:vertical; }
 button { margin-top:10px; padding:9px 16px; border:0; border-radius:6px; background:var(--teal); color:#fff; font:inherit; cursor:pointer; }
 button:disabled { opacity:.5; cursor:wait; }
 .row { display:flex; gap:8px; align-items:center; margin-top:10px; font-size:14px; }
 .setups { display:grid; gap:12px; } .setup h3 { margin:6px 0 0; font-size:15px; }
 .results { display:grid; gap:16px; } .results.two { grid-template-columns:1fr 1fr; }
 @media (max-width: 1100px) { .results.two { grid-template-columns:1fr; } }
 .tag { display:inline-block; font-size:12px; background:var(--teal-bg); color:var(--teal); border-radius:999px; padding:2px 9px; margin:0 4px 4px 0; }
 .answer { white-space:pre-wrap; background:#fff; border-left:4px solid var(--teal); padding:10px 12px; margin:10px 0; }
 .cite { color:var(--teal); font-weight:600; cursor:pointer; }
 details { border-top:1px solid var(--line); padding:8px 0; } summary { cursor:pointer; }
 .passage.hl { background:#fff6d6; }
 .meta { color:var(--soft); font-size:13px; } pre { white-space:pre-wrap; font-size:13px; background:#fff; padding:10px; border-radius:6px; }
 .err { color:var(--warn); white-space:pre-wrap; }
 .hist { margin-top:16px; font-size:14px; } .hist a { display:block; color:var(--teal); cursor:pointer; margin:3px 0; }
</style></head><body>
<header><h1>Guideline assistant</h1><p>Answers from six Ethiopian public health guidelines. For learning only: not for patient care.</p></header>
<main>
<aside>
 <label for="q">Your question</label>
 <textarea id="q" placeholder="e.g. What is the treatment for a 7-year-old with cholera who has stopped vomiting?"></textarea>
 <div class="row"><input type="checkbox" id="searchonly"><label for="searchonly" style="margin:0">Search only (no LLM, no key needed)</label></div>
 <div class="row"><input type="checkbox" id="compare"><label for="compare" style="margin:0">Compare two setups</label></div>
 <div class="setups" id="setups"></div>
 <button id="go">Ask</button>
 <div class="hist" id="hist"></div>
</aside>
<section><div id="out" class="results"><p class="meta">Ask a question to see the answer, the passages it came from, and the prompt.</p></div></section>
</main>
<script>
const MODEL = "{{MODEL}}";
function setupForm(i) {
  return `<div class="setup" id="setup${i}"><h3>Setup ${i ? "B" : "A"}</h3>
   <label>Text</label><select data-f="docs"><option value="pdf">PDF text (pypdf)</option><option value="markdown">Markdown (LlamaParse)</option></select>
   <label>Chunking</label><select data-f="chunking"><option value="fixed">fixed: 200-word windows</option><option value="structure">structure: by headings (Markdown only)</option></select>
   <label>Search</label><select data-f="retriever"><option value="tfidf">keywords (TF-IDF)</option><option value="embed">meaning (embeddings)</option></select>
   <label>Passages given to the model (k)</label><input type="number" data-f="k" value="5" min="1" max="15">
   <label>Model</label><select data-f="model"><option>${MODEL}</option><option>qwen/qwen3.5-9b</option></select></div>`;
}
const setups = document.getElementById("setups");
function drawSetups() { setups.innerHTML = setupForm(0) + (compare.checked ? setupForm(1) : ""); }
const compare = document.getElementById("compare"); compare.onchange = drawSetups; drawSetups();
function readSetup(i) { const o = {}; document.querySelectorAll(`#setup${i} [data-f]`).forEach(e => o[e.dataset.f] = e.type === "number" ? +e.value : e.value); return o; }
function esc(s) { return (s || "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function render(r, col) {
  const s = r.setup;
  const cites = esc(r.answer || "").replace(/\\[(\\d+)\\]/g, (m, n) => `<span class="cite" onclick="hl(${col},${n})">[${n}]</span>`);
  const passages = r.passages.map(p => `<details class="passage" id="p${col}_${p.n}"><summary><b>[${p.n}]</b> ${esc(p.doc)}, ${p.pages} <span class="meta">(score ${p.score})</span></summary><pre>${esc(p.text)}</pre></details>`).join("");
  return `<div class="result"><div>${["docs","chunking","retriever"].map(f => `<span class="tag">${s[f]}</span>`).join("")}<span class="tag">k=${s.k}</span><span class="tag">${esc(s.model)}</span></div>
   ${r.answer === null ? '<p class="meta">Search only: no answer generated.</p>' : `<div class="answer">${cites}</div>`}
   <p class="meta">${r.seconds} s${r.tokens ? ", " + r.tokens + " tokens" : ""}</p>
   <h3 style="font-size:15px;margin:8px 0">Passages retrieved</h3>${passages}
   <details><summary>The exact prompt sent to the model</summary><pre>${esc(r.prompt)}</pre></details></div>`;
}
function hl(col, n) { const el = document.getElementById(`p${col}_${n}`); if (!el) return; el.open = true; el.classList.add("hl"); el.scrollIntoView({behavior:"smooth", block:"center"}); setTimeout(() => el.classList.remove("hl"), 2000); }
const history = [];
async function go() {
  const q = document.getElementById("q").value.trim(); if (!q) return;
  const btn = document.getElementById("go"); btn.disabled = true; btn.textContent = "Working…";
  const out = document.getElementById("out");
  const body = {question: q, search_only: document.getElementById("searchonly").checked, setups: [readSetup(0)].concat(compare.checked ? [readSetup(1)] : [])};
  if (body.setups.some(s => s.retriever === "embed")) out.innerHTML = '<p class="meta">Embedding search: the first time for a text and chunking, every passage is embedded (a few minutes, once).</p>';
  try {
    const res = await (await fetch("/ask", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)})).json();
    if (res.error) { out.innerHTML = `<p class="err">${esc(res.error)}</p>`; }
    else { out.className = "results" + (res.results.length > 1 ? " two" : ""); out.innerHTML = res.results.map(render).join("");
           if (!history.includes(q)) { history.unshift(q); document.getElementById("hist").innerHTML = "<b>Asked</b>" + history.map(h => `<a onclick="document.getElementById('q').value=this.textContent">${esc(h)}</a>`).join(""); } }
  } catch (e) { out.innerHTML = `<p class="err">${esc(String(e))}</p>`; }
  btn.disabled = false; btn.textContent = "Ask";
}
document.getElementById("go").onclick = go;
document.getElementById("q").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) go(); });
</script></body></html>"""

if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in your browser (Ctrl+C to stop).")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
