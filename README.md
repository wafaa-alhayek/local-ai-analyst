# Describe Local AI Analyst

Ask your data questions in **Arabic or English** and get the answer, a chart, and a plain-language explanation. Everything runs **on your own computer**. An open-source AI model writes the SQL, a local DuckDB database runs it, and no data is ever sent to the cloud.

> "ما هي المنطقة التي وصلها أقل عدد من الأسر في شهر مارس؟"
> → SQL → table → chart → "وصلت رفح إلى أقل عدد من الأسر في مارس…"

## Why it matters

- **Private by design.** Client data such as survey answers or beneficiary lists never leaves the machine. Nothing is sent to OpenAI or Google, and there are no API bills.
- **Bilingual.** Questions, data values and answers can be in Arabic or English.
- **Shows its work.** Every answer includes the SQL it ran, so an analyst can check it.
- **Fixes its own mistakes.** If a query fails, the error goes back to the model, which corrects the query (up to 2 retries).
- **Safe.** Only read-only `SELECT` queries are allowed. Anything that could change or export data is blocked.
- **Forecasts.** Ask "forecast … for the next 3 months" and [Google TimesFM 2.5](https://github.com/google-research/timesfm), a 200M-parameter open forecasting model (Apache-2.0), predicts it locally and shows an 80% likely range.

## How it works

```
your files (CSV / Excel / Parquet / JSON)
        │
        ▼
 DuckDB (in memory) ──► schema + sample rows ──► local LLM (Ollama) ──► SQL
        ▲                                                               │
        └────────── run (read-only), retry with the error if it fails ◄─┘
                                   │
                                   ▼
                    table + auto chart + explanation (local LLM)

 forecast question? the SQL returns the history ──► TimesFM 2.5 ──► forecast + 80% range
```

| Part | Tool | Why |
|---|---|---|
| AI model | Qwen2.5-Coder 7B through [Ollama](https://ollama.com) | Open source and strong at SQL, handles Arabic, and runs on a 16 GB laptop |
| Data engine | DuckDB | Free, fast, a single file, and reads CSV/Excel directly |
| Forecasting | Google TimesFM 2.5 (200M) | Open (Apache-2.0, commercial use OK), runs on CPU, zero-shot |
| App | Streamlit + Altair | A web page that runs locally |

Lighter machines can use `qwen2.5:3b` (8 GB RAM). Pick the model in the sidebar.

## Quick start

See [docs/SETUP.md](docs/SETUP.md) for the step-by-step guide.

```bash
ollama pull qwen2.5-coder:7b
pip install -r requirements.txt
pip install -r requirements-forecast.txt   # optional: TimesFM forecasting
streamlit run app.py
```

Command line:

```bash
python ask.py "Which region reached the most households?" data/aid_distributions.csv
```

## Sample data

- `data/aid_distributions.csv` contains 600 **synthetic** aid distributions by region, partner and item for Jan–Sep 2026. Rafah has a deliberate dip in March to give the AI something to find.
- `data/survey_responses.csv` contains 40 bilingual sample survey answers from the survey analyzer project.

## Project layout

```
analyst/engine.py   core: data loading, prompts, SQL guard, retry loop, chart choice
analyst/forecast.py TimesFM 2.5 wrapper, horizon parsing, baseline fallback
app.py              Streamlit app
ask.py              command-line version
tests/              runs without a model (fake LLM)
```

Run the tests with `pytest -q`.

## Roadmap

1. Connectors for Snowflake and BigQuery, so the same questions work on cloud data when a client wants it.
2. A "business glossary" file that teaches the model client terms, for example "beneficiary = household".
3. Follow-up questions that remember the previous answer.
4. Scheduled reports: a saved list of questions that runs weekly and exports a PDF brief.
