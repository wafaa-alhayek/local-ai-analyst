"""Describe Local AI Analyst: ask your data questions in Arabic or English.

Run:  streamlit run app.py
Needs Ollama running locally with the chosen model pulled (see docs/SETUP.md).
"""

import tempfile
from pathlib import Path

import altair as alt
import streamlit as st

from analyst import Analyst, OllamaClient, Workspace

DATA_DIR = Path(__file__).parent / "data"
MODELS = ["qwen2.5-coder:7b", "qwen2.5:7b", "qwen2.5:3b", "llama3.1:8b"]
EXAMPLES = [
    "Which region reached the most households in total?",
    "Show monthly households reached by region",
    "ما هي المنطقة التي وصلها أقل عدد من الأسر في شهر مارس؟",
    "How many survey responses came from each region and age group?",
]

st.set_page_config(page_title="Local AI Analyst", page_icon="📊", layout="wide")


@st.cache_resource
def get_workspace() -> Workspace:
    ws = Workspace()
    for f in sorted(DATA_DIR.glob("*.csv")):
        ws.add_file(f)
    return ws


ws = get_workspace()

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Local model", MODELS, help="Runs on this computer through Ollama.")
    llm = OllamaClient(model=model)
    if llm.available():
        st.success(f"{model} is ready. Your data stays on this machine.")
    else:
        st.error(f"Can't reach {model}. Start Ollama and run `ollama pull {model}`.")

    st.header("Your data")
    uploads = st.file_uploader("Add CSV, Excel, Parquet or JSON",
                               type=["csv", "xlsx", "xls", "parquet", "json"],
                               accept_multiple_files=True)
    for up in uploads or []:
        tmp = Path(tempfile.gettempdir()) / up.name
        tmp.write_bytes(up.getvalue())
        ws.add_file(tmp)
    for t in ws.tables:
        n = ws.con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        with st.expander(f"{t} ({n:,} rows)"):
            st.dataframe(ws.con.execute(f'SELECT * FROM "{t}" LIMIT 20').df(), hide_index=True)

st.title("📊 Local AI Analyst")
st.caption("Ask a question about your data in Arabic or English. "
           "An open-source model on this computer writes the query, runs it and explains the answer.")

if "history" not in st.session_state:
    st.session_state.history = []

cols = st.columns(len(EXAMPLES))
picked = None
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex, width="stretch"):
        picked = ex

question = st.chat_input("Ask a question about your data…") or picked


def render_chart(ans):
    spec, df = ans.chart, ans.data
    if not spec:
        return
    enc = {"x": alt.X(spec["x"], sort=None if spec["type"] != "bar" else "-y"),
           "y": alt.Y(spec["y"]),
           "tooltip": list(df.columns)}
    if spec.get("color"):
        enc["color"] = alt.Color(spec["color"])
    mark = {"line": alt.Chart(df).mark_line(point=True),
            "bar": alt.Chart(df).mark_bar(),
            "scatter": alt.Chart(df).mark_circle(size=60)}[spec["type"]]
    st.altair_chart(mark.encode(**enc).properties(height=360), width="stretch")


if question:
    with st.spinner("Thinking locally…"):
        try:
            ans = Analyst(ws, llm).ask(question)
        except Exception as e:
            st.error(f"The local model could not be reached: {e}")
            ans = None
    if ans:
        st.session_state.history.insert(0, ans)

for ans in st.session_state.history:
    with st.container(border=True):
        st.markdown(f"**{ans.question}**")
        if ans.error:
            st.error(f"I couldn't answer this. Last error: {ans.error}")
            st.code(ans.sql, language="sql")
            continue
        if ans.explanation:
            st.markdown(ans.explanation)
        render_chart(ans)
        tab_data, tab_sql = st.tabs(["Data", "SQL"])
        tab_data.dataframe(ans.data, hide_index=True)
        tab_sql.code(ans.sql, language="sql")
        if len(ans.attempts) > 1:
            tab_sql.caption(f"Fixed itself after {len(ans.attempts) - 1} failed attempt(s).")
