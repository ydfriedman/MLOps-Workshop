# Guideline sources and the Markdown versions

The assistant reads six Ethiopian public health guidelines. The PDFs are downloaded from their publishers by
`python scripts/download.py guidelines` (URLs and checksums in [`scripts/downloads.json`](../../scripts/downloads.json))
and are not stored in this repository. Their Markdown versions, in [`../guidelines_md/`](../guidelines_md/), are.

| File stem | Document | Pages | Source URL |
|---|---|---|---|
| `ephi_cholera_2022` | EPHI: National Guideline for Cholera Surveillance and Outbreak Response (2022) | 124 | https://ephi.gov.et/wp-content/uploads/2022/11/Guideline_for_Cholera_Surveillance_and_Outbreak_Response_Final_V2.pdf |
| `ephi_emergency_nutrition_2022` | EPHI: Emergency Nutrition Guideline (2022) | 92 | https://ephi.gov.et/wp-content/uploads/2022/03/EPHI-Emergency-Nutrition-GuidelineOR.pdf |
| `ephi_climate_sensitive_diseases_2022` | EPHI: Climate-Sensitive Diseases Surveillance Manual (2022) | 110 | https://ephi.gov.et/wp-content/uploads/2022/07/EPHI_cPHEM_EWISMD_Climate_Sensitive_Diseases_Surveillance_Manual_2022.pdf |
| `ephi_cebs_2021` | EPHI: Community and Event-Based Surveillance Implementation Manual (2021) | 50 | https://ephi.gov.et/wp-content/uploads/2022/07/EPHI_cPHEM_EWISMD_CEBS_Implementation_Manual_2021.pdf |
| `moh_acute_malnutrition_2019` | Ministry of Health: National Guideline for the Management of Acute Malnutrition (May 2019; copy hosted by the Global Nutrition Cluster) | 156 | https://stg.nutritioncluster.net/sites/nutritioncluster.com/files/2022-06/National%20Guideline%20for%20Management%20of%20Acute%20Malnutrition%20May%202019%20Version.pdf |
| `moh_tb_2017` | Ministry of Health: National Guidelines for TB, DR-TB and Leprosy, 6th edition (2017; copy hosted by WHO AFRO) | 217 | https://afro.who.int/sites/default/files/2019-04/Ethiopia-National%20guidelines%20for%20TB%2C%20DR-TB%20and%20Leprosy%20in%20Ethiopia%20-%20Sixth%20Edition.pdf |

## Licence

None of the six documents carries a licence or copyright statement (checked in the full text). They are public
guidance from the Ethiopian Public Health Institute and the Ministry of Health, published on the open web for
health workers. **The Markdown files are a full-text copy of them**, which goes further than linking to the PDFs:
confirm with EPHI / the Ministry, or keep the repository private, before publishing it. Each file names its source
document; the text is the publishers' and is reproduced unchanged apart from layout.

## How the Markdown was made

Parsed with **LlamaParse** (LlamaIndex Cloud, v2 parse API, Python SDK `llama-cloud` 2.19) on **2026-10-21**, once,
by [`../facilitator/parse_guidelines.py`](../facilitator/parse_guidelines.py). Exact request parameters and job ids:
[`../guidelines_md/parse_log.json`](../guidelines_md/parse_log.json).

- **Tier `agentic`, version `2026-10-16`** (pinned). 10 credits/page; 749 pages = 7,480 credits, plus 420 for the
  pilots. `agentic_plus` (45/page, ~33,700 credits) does not fit the free plan's 20,000 a month. On the 6 pilot
  pages both tiers gave identical dosing numbers; agentic_plus only joined the F-75 table's two header rows, which
  the script's post-processing now does.
- **Custom prompt**: exact table rows and numbers, one header row per table, flowcharts as nested lists, charts as
  tables of values, heading levels kept, nothing added.
- **Output**: Markdown tables (`output_tables_as_markdown`, `compact_markdown_tables`), tables not merged across
  pages (`merge_continued_tables: false`, so every table stays on its page), `inline_images`, OCR language `en`,
  `extract_printed_page_number`; `expand=["markdown", "metadata"]`.
- **Pages**: one block per PDF page, opened by `<!-- page N -->`, where N is the **PDF page index** (1-based, what
  rag.py cites). Where the printed page number differs it is kept: `<!-- page 42 (printed 29) -->`.
- **Headers, footers, page numbers**: LlamaParse returns them apart from the page body. Page numbers and running
  heads (lines on 3+ pages) are dropped; what is left is kept at the end of its page: footnotes (6 pages in the
  cholera guideline, 4 in emergency nutrition) and one chapter heading (CEBS p18) taken for a running head.
- **Post-processing** (the script's `clean()`): image links dropped; two-row table headers that came back as a data
  row joined into the header; highlight marks (`<mark>`, `~~`) around cross-references in the acute malnutrition
  PDF removed; over-escaped `\\\*` normalised.

## Quality checks (2026-10-21)

**Tables checked by eye against the PDF**, cell by cell, all exact: cholera p29 (Plan C Ringer's lactate, age
bands), p38 (antibiotics, two-column page); acute malnutrition p41 (OTP routine medicines), p58 (vitamin A by age),
p87 (TSFP medicines), p120 (F-75 volumes, 32 weight bands × 5 columns); emergency nutrition p52 (vitamin A
prophylaxis); TB p56 (INH weight bands), p205 (Annex 6: 21 drugs × 5–6 weight bands). Flowcharts (acute
malnutrition p25, TB p42) come out with every box's text; the arrows are mostly right (the pilot of p25 had one
extra arrow, the full run did not).

**Known imperfections**: footnote letters run into the text ("Volume of F-75 per Feed (ml)a"); rows that span a
table (bedaquiline in Annex 6) sit in the first value column; a heading carries over to the next page when that page
has none of its own (acute malnutrition p25 is under "Hospital Level" from p24). The documents contain no Amharic
text.

## Chunking, PDF vs Markdown (2026-10-21)

Two ways to cut the text into passages, set by `--chunking` (rag.py):

- **fixed**: overlapping windows of 200 words (40 overlap) within one page, as the lesson does with the PDFs.
  On the Markdown, the same windows run over each page's Markdown text.
- **structure** (Markdown only): passages follow the headings, across page breaks. A section that fits in 200
  words is one passage; a longer one is cut at paragraph breaks; small sections are packed with the next. A table
  is never cut, and a table continued on the next page is first joined into one (27 joins: the F-75 and F-100
  annexes, glossaries, the TB adverse-event grading table, among others; a continuation whose header is
  reworded is left as a separate table). Each passage starts with its heading path
  ("4. CHOLERA OUTBREAK RESPONSE > 4.2. Case Management > Plan C: ..."), built from the printed section numbers
  because LlamaParse's heading levels are unreliable. Each passage has a page range (`page`..`end_page`, PDF page
  index); citations show it ("pages 28–29") and a retrieval hit counts if the expected page is in the range.
  417 of 1,187 passages span more than one page; mean 189 words, 19 over 400 (whole tables, up to 2,167 words
  for the F-75 annex: the embedding model reads only its first ~512 tokens).

How the choices compare on the lesson's own questions and cases: notebooks 3 and 4, run live.
