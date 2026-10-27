"""Tests that run without a model: a fake LLM returns scripted replies."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyst import Analyst, Workspace  # noqa: E402
from analyst.engine import check_read_only, detect_language, extract_sql  # noqa: E402


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, system, user, temperature=0.0):
        self.prompts.append(user)
        return self.replies.pop(0)


@pytest.fixture
def ws():
    w = Workspace()
    for f in sorted((ROOT / "data").glob("*.csv")):
        w.add_file(f)
    return w


def test_tables_loaded(ws):
    assert ws.tables == ["aid_distributions", "survey_responses"]
    assert "households_reached" in ws.schema_text()


def test_read_only_guard():
    check_read_only("SELECT 1")
    check_read_only("WITH a AS (SELECT 1) SELECT * FROM a")
    check_read_only("SELECT 'drop' AS word")
    for bad in ["DROP TABLE x", "SELECT 1; DELETE FROM x", "COPY x TO 'f.csv'",
                "SELECT * FROM x; ATTACH 'y'"]:
        with pytest.raises(ValueError):
            check_read_only(bad)


def test_ask_bar_chart(ws):
    llm = FakeLLM([
        '```sql\nSELECT region, SUM(households_reached) AS households '
        'FROM aid_distributions GROUP BY region ORDER BY households DESC\n```',
        "Gaza City reached the most households.",
    ])
    ans = Analyst(ws, llm).ask("Which region reached the most households?")
    assert not ans.error
    assert len(ans.data) == 5
    assert ans.chart["type"] == "bar"
    assert ans.explanation.startswith("Gaza City")


def test_self_repair(ws):
    llm = FakeLLM([
        "SELECT regionn FROM aid_distributions",
        "SELECT date_trunc('month', CAST(\"date\" AS DATE)) AS month, "
        "SUM(households_reached) AS hh FROM aid_distributions GROUP BY 1 ORDER BY 1",
        "Monthly totals are steady.",
    ])
    ans = Analyst(ws, llm).ask("Monthly households?")
    assert len(ans.attempts) == 2 and ans.attempts[0]["error"]
    assert "previous query failed" in llm.prompts[1]
    assert ans.chart["type"] == "line"


def test_gives_up_after_retries(ws):
    llm = FakeLLM(["SELECT nope FROM aid_distributions"] * 3)
    ans = Analyst(ws, llm).ask("broken")
    assert ans.error and len(ans.attempts) == 3


def test_helpers():
    assert detect_language("ما هي المنطقة") == "Arabic"
    assert detect_language("Which region") == "English"
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


# ---------------------------------------------------------------- forecasting

from analyst.forecast import Forecaster, is_forecast_question, parse_horizon  # noqa: E402


class FakeForecaster(Forecaster):
    """Baseline backend, so tests never download TimesFM."""

    def _load(self):
        self.backend = "baseline"


MONTHLY_SQL = ("SELECT date_trunc('month', CAST(\"date\" AS DATE)) AS period, "
               "SUM(households_reached) AS households FROM aid_distributions GROUP BY 1 ORDER BY 1")


def test_forecast_detection_and_horizon():
    assert is_forecast_question("Forecast households for the next 3 months")
    assert is_forecast_question("توقع عدد الأسر في الأشهر القادمة")
    assert not is_forecast_question("Which region reached the most households?")
    assert parse_horizon("next 6 months") == 6
    assert parse_horizon("الأشهر ٤ أشهر القادمة") == 4
    assert parse_horizon("next three weeks") == 3
    assert parse_horizon("forecast it") == 3


def test_forecast_single_series(ws):
    llm = FakeLLM([f"```sql\n{MONTHLY_SQL}\n```", "Expect about 9,000 households a month."])
    ans = Analyst(ws, llm, FakeForecaster()).ask("Forecast households for the next 3 months")
    assert not ans.error, ans.error
    fc = ans.data[ans.data.kind == "forecast"]
    assert len(fc) == 3 and (fc.low <= fc.value).all() and (fc.value <= fc.high).all()
    assert fc.period.min() > ans.data[ans.data.kind == "actual"].period.max()
    assert ans.chart["type"] == "forecast" and ans.chart["color"] is None
    assert ans.forecast_backend == "baseline"


def test_forecast_by_group(ws):
    sql = MONTHLY_SQL.replace("AS period, ", "AS period, region, ").replace("GROUP BY 1", "GROUP BY 1, 2")
    llm = FakeLLM([sql, "ok"])
    ans = Analyst(ws, llm, FakeForecaster()).ask("Predict next 2 months by region")
    assert not ans.error, ans.error
    assert ans.chart["color"] == "region"
    assert len(ans.data[ans.data.kind == "forecast"]) == 5 * 2
