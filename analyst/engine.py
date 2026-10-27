"""Local natural-language data analyst.

Flow: load files into DuckDB -> describe schema -> local LLM writes SQL ->
validate (read-only) -> run, retrying with the error if it fails ->
pick a chart -> local LLM explains the result in the user's language.

Nothing leaves the machine: the model runs in Ollama on localhost and the
data sits in an in-memory DuckDB database.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from .forecast import Forecaster, asks_by_group, forecast_frame, is_forecast_question, parse_horizon

DEFAULT_MODEL = os.environ.get("ANALYST_MODEL", "qwen2.5-coder:7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_RETRIES = 2
SAMPLE_ROWS = 3

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|"
    r"import|install|load|pragma|set|call|checkpoint|vacuum)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- LLM client

class OllamaClient:
    """Minimal Ollama chat client using only the standard library."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["message"]["content"]

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.loads(r.read())["models"]]
            return any(n == self.model or n.startswith(self.model + ":") for n in names)
        except Exception:
            return False


# ---------------------------------------------------------------- data layer

def table_name(path: str | Path) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]", "_", Path(path).stem).lower()
    return name if not name[0].isdigit() else f"t_{name}"


class Workspace:
    """An in-memory DuckDB database holding the user's files as tables."""

    def __init__(self):
        self.con = duckdb.connect(":memory:")
        self.tables: list[str] = []

    def add_file(self, path: str | Path, name: str | None = None) -> str:
        path = Path(path)
        name = name or table_name(path)
        ext = path.suffix.lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
            self.con.register("_tmp", df)
            self.con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _tmp')
            self.con.unregister("_tmp")
        elif ext == ".parquet":
            self.con.execute(f"CREATE OR REPLACE TABLE \"{name}\" AS SELECT * FROM read_parquet(?)", [str(path)])
        elif ext == ".json":
            self.con.execute(f"CREATE OR REPLACE TABLE \"{name}\" AS SELECT * FROM read_json_auto(?)", [str(path)])
        else:
            self.con.execute(f"CREATE OR REPLACE TABLE \"{name}\" AS SELECT * FROM read_csv_auto(?)", [str(path)])
        if name not in self.tables:
            self.tables.append(name)
        return name

    def add_dataframe(self, df: pd.DataFrame, name: str) -> str:
        self.con.register("_tmp", df)
        self.con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _tmp')
        self.con.unregister("_tmp")
        if name not in self.tables:
            self.tables.append(name)
        return name

    def schema_text(self) -> str:
        """Compact schema + a few sample rows, which is what the model sees."""
        parts = []
        for t in self.tables:
            cols = self.con.execute(f'DESCRIBE "{t}"').fetchall()
            n = self.con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            col_lines = []
            for c in cols:
                cname, ctype = c[0], c[1]
                hint = ""
                if ctype == "VARCHAR":
                    distinct = self.con.execute(
                        f'SELECT COUNT(DISTINCT "{cname}") FROM "{t}"').fetchone()[0]
                    if distinct <= 12:
                        vals = [r[0] for r in self.con.execute(
                            f'SELECT DISTINCT "{cname}" FROM "{t}" WHERE "{cname}" IS NOT NULL LIMIT 12').fetchall()]
                        hint = f"  -- values: {vals}"
                col_lines.append(f'  "{cname}" {ctype}{hint}')
            sample = self.con.execute(f'SELECT * FROM "{t}" LIMIT {SAMPLE_ROWS}').df()
            sample = sample.map(lambda v: v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v)
            parts.append(
                f'TABLE "{t}" ({n} rows)\n' + "\n".join(col_lines)
                + "\nSample rows:\n" + sample.to_csv(index=False)
            )
        return "\n".join(parts)

    def run(self, sql: str) -> pd.DataFrame:
        check_read_only(sql)
        return self.con.execute(sql).df()


def check_read_only(sql: str) -> None:
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise ValueError("Only one statement is allowed.")
    if not re.match(r"^\s*(with|select)\b", stripped, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    if FORBIDDEN.search(re.sub(r"'[^']*'", "''", stripped)):
        raise ValueError("Query contains a forbidden keyword.")


# ---------------------------------------------------------------- prompts

SQL_SYSTEM = """You are a senior data analyst who writes DuckDB SQL.
Rules:
- Return ONLY one SQL SELECT query inside a ```sql block. No other text.
- Use only the tables and columns in the schema. Quote column names with double quotes.
- Aggregate when the question asks about totals, trends or comparisons; order results sensibly.
- For trends over time use date_trunc('month', CAST("date" AS DATE)) or similar.
- To pick one month, filter with date_trunc('month', CAST("date" AS DATE)) = DATE '2026-03-01'.
  Never compare a date with a partial string such as '2026-03'.
- Never filter on CURRENT_DATE or now(): use the dates in the data.
- Never add LIMIT to an aggregated result. For "which ... most / least" questions
  return every group, ordered, so they can be compared.
- Limit raw row listings (no aggregation) to 50 rows.
- The question may be in Arabic or English; data values may be in either language.
"""

FORECAST_RULES = """
This is a FORECAST question. Do NOT predict the future in SQL. Return only the
past values as a time series that a forecasting model will extend:
- first column: the period, e.g. date_trunc('month', CAST("date" AS DATE)) AS period
- optional second column: ONE category, ONLY if the question says "by", "each"
  or "per" a category (e.g. "by region"); otherwise sum everything per period
- use ALL the history in the table: no date filters, no LIMIT
- last column: the numeric measure, aggregated per period
- ORDER BY the period. Choose month unless the question implies days or weeks.
"""

EXPLAIN_SYSTEM = """You are a data analyst writing for a non-technical client.
Given a question and FACTS computed from the query result, write 2-3 short
sentences that answer the question. Use only numbers that appear in FACTS,
copied exactly; never add, average or invent numbers. Mention anything that
stands out. Reply in {language}.
"""


def facts(df: pd.DataFrame, forecasting: bool = False, max_items: int = 12) -> str:
    """Plain-text facts the explanation must stick to (small models invent numbers
    when asked to read a whole table)."""
    def fmt(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}".rstrip("0").rstrip(".")
        if isinstance(v, pd.Timestamp):
            return f"{v:%B %Y}"
        return str(v)

    lines = []
    if forecasting:
        group = next((c for c in df.columns if c not in ("period", "value", "low", "high", "kind")), None)
        for key, g in (df.groupby(group, sort=True) if group else [(None, df)]):
            name = f"{key}: " if group else ""
            act, fc = g[g.kind == "actual"], g[g.kind == "forecast"]
            last = act.sort_values("period").iloc[-1]
            lines.append(f"{name}last actual month {fmt(last.period)} = {fmt(last.value)}; "
                         f"average of the actual months = {fmt(act.value.mean())}")
            for r in fc.itertuples():
                lines.append(f"{name}FORECAST {fmt(r.period)} = {fmt(r.value)} "
                             f"(likely between {fmt(r.low)} and {fmt(r.high)})")
        return "\n".join(lines[:max_items * 2])

    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num or df.empty:
        return df.head(max_items).to_csv(index=False)
    y = num[-1]
    label_cols = [c for c in df.columns if c != y and c not in num[:-1]] or [c for c in df.columns if c != y]
    label = lambda r: ", ".join(fmt(r[c]) for c in label_cols)  # noqa: E731
    lines.append(f"{len(df)} rows; measure = {y}")
    ranked = df.sort_values(y, ascending=False)
    hi, lo = ranked.iloc[0], ranked.iloc[-1]
    lines.append(f"highest: {label(hi)} = {fmt(hi[y])}")
    if len(df) > 1:
        lines.append(f"lowest: {label(lo)} = {fmt(lo[y])}")
    if len(df) <= max_items:
        lines += [f"{label(r)} = {fmt(r[y])}" for _, r in ranked.iterrows()]
    return "\n".join(lines)


def drop_aggregate_limit(sql: str, question: str) -> str:
    """Small models add LIMIT 1 to 'which region most?' and lose the comparison.
    Keep a LIMIT only when the question asks for a number of results ('top 3')."""
    if re.search(r"\bgroup\s+by\b", sql, re.I) and not re.search(r"\d|\btop\b", question, re.I):
        return re.sub(r"\s+limit\s+\d+\s*$", "", sql, flags=re.I)
    return sql


def detect_language(text: str) -> str:
    return "Arabic" if re.search(r"[؀-ۿ]", text) else "English"


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    sql = (m.group(1) if m else text).strip()
    return sql.rstrip(";").strip()


# ---------------------------------------------------------------- charts

def suggest_chart(df: pd.DataFrame) -> dict | None:
    """Pick a sensible chart from the result's shape. Returns an Altair-ready spec."""
    if df is None or df.empty or len(df.columns) < 2 or len(df) < 2:
        return None
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dt = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat = [c for c in df.columns if c not in num and c not in dt]
    if not num:
        return None
    y = num[-1] if (dt or cat) else num[1] if len(num) > 1 else None
    if dt:
        color = cat[0] if cat else None
        return {"type": "line", "x": dt[0], "y": y, "color": color}
    if cat:
        color = cat[1] if len(cat) > 1 else None
        return {"type": "bar", "x": cat[0], "y": y, "color": color}
    if len(num) >= 2:
        return {"type": "scatter", "x": num[0], "y": num[1], "color": None}
    return None


# ---------------------------------------------------------------- analyst

@dataclass
class Answer:
    question: str
    sql: str = ""
    data: pd.DataFrame | None = None
    chart: dict | None = None
    explanation: str = ""
    error: str = ""
    attempts: list[dict] = field(default_factory=list)
    forecast_backend: str = ""  # "timesfm" or "baseline" when this is a forecast


class Analyst:
    def __init__(self, workspace: Workspace, llm=None, forecaster: Forecaster | None = None):
        self.ws = workspace
        self.llm = llm or OllamaClient()
        self.forecaster = forecaster or Forecaster()

    def ask(self, question: str, explain: bool = True) -> Answer:
        ans = Answer(question=question)
        forecasting = is_forecast_question(question)
        system = SQL_SYSTEM + (FORECAST_RULES if forecasting else "")
        schema = self.ws.schema_text()
        prompt = f"Schema:\n{schema}\nQuestion: {question}"
        for attempt in range(MAX_RETRIES + 1):
            sql = drop_aggregate_limit(extract_sql(self.llm.chat(system, prompt)), question)
            try:
                df = self.ws.run(sql)
                if forecasting:
                    df = forecast_frame(df, parse_horizon(question), self.forecaster,
                                        keep_groups=asks_by_group(question))
                ans.sql, ans.data = sql, df
                ans.attempts.append({"sql": sql, "error": None})
                break
            except Exception as e:  # feed the error back so the model can fix it
                ans.attempts.append({"sql": sql, "error": str(e)})
                prompt = (
                    f"Schema:\n{schema}\nQuestion: {question}\n\n"
                    f"Your previous query failed.\nQuery:\n{sql}\nError: {e}\n"
                    "Write a corrected query."
                )
        else:
            ans.sql = ans.attempts[-1]["sql"]
            ans.error = ans.attempts[-1]["error"]
            return ans

        if forecasting:
            ans.forecast_backend = self.forecaster.backend
            ans.chart = {"type": "forecast", "x": "period", "y": "value",
                         "color": next((c for c in ans.data.columns
                                        if c not in ("period", "value", "low", "high", "kind")), None)}
        else:
            ans.chart = suggest_chart(ans.data)
        if explain:
            note = ""
            if forecasting:
                note = ("\nFORECAST lines are predictions by the "
                        f"{'TimesFM model' if ans.forecast_backend == 'timesfm' else 'baseline model'}, "
                        "not actual data; say they are forecasts and give the likely range.")
            ans.explanation = self.llm.chat(
                EXPLAIN_SYSTEM.format(language=detect_language(question)),
                f"Question: {question}\nFACTS:\n{facts(ans.data, forecasting)}{note}",
                temperature=0.2,
            ).strip()
        return ans
