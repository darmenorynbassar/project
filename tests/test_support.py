"""Unit tests for the pipeline and support layer (no language model needed).

Usage:
    pytest tests/test_support.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import chatbot  # noqa: E402
import features  # noqa: E402
import support  # noqa: E402

WEEK = 8


def daily_clicks():
    return pd.DataFrame({"code_module": "AAA", "code_presentation": "2014J", "id_student": 1,
                         "date": [-3, 0, 10, 55, 56, 90], "activity_type": "resource",
                         "sum_click": [1, 2, 3, 4, 100, 100]})


def test_vle_features_ignore_clicks_after_cutoff():
    daily = daily_clicks()
    before = features.vle_features(daily, 56)
    changed = daily.copy()
    changed.loc[changed["date"] >= 56, "sum_click"] = 9999
    pd.testing.assert_frame_equal(before, features.vle_features(changed, 56))
    assert before["clicks_total"].iloc[0] == 10


def test_no_student_unregistered_before_cutoff():
    reg = pd.read_csv(features.data_dir() / "studentRegistration.csv")
    early = reg[reg["date_unregistration"] < 7 * WEEK]
    kept = pd.read_parquet(features.PROCESSED / f"features_week{WEEK}.parquet")
    assert kept.merge(early, on=features.KEYS).empty


def test_support_students_are_unseen_2014():
    kept = pd.read_parquet(features.PROCESSED / f"features_week{WEEK}.parquet")
    seen = set(kept[kept["year"] == 2013]["id_student"])
    df = support.load(WEEK)
    assert df["code_presentation"].str.startswith("2014").all()
    assert not seen & set(df["id_student"])


def test_risk_list_is_flagged_and_sorted():
    result = support.risk_list(WEEK, "BBB", limit=20)
    scores = [s["risk_score"] for s in result["students"]]
    assert scores == sorted(scores, reverse=True)
    assert {s["code_module"] for s in result["students"]} == {"BBB"}
    assert {s["risk_level"] for s in result["students"]} <= {"High", "Medium"}


def test_unknown_student():
    assert support.explain(999999, WEEK)["found"] is False
    assert support.draft_letter(999999, WEEK)["found"] is False


def test_letter_contains_student_and_disclaimer():
    df = support.load(WEEK)
    student = df[~df["id_student"].duplicated(keep=False) & df["flagged"]].iloc[0]
    letter = support.draft_letter(student["id_student"], WEEK)["letter"]
    assert str(student["id_student"]) in letter
    assert "decision support" in letter


def test_no_demographics_reach_the_support_layer():
    columns = set(support.load(WEEK).columns)
    assert not columns & set(features.DEMOGRAPHIC_COLS)


def test_chatbot_has_no_action_tools():
    names = {t["function"]["name"] for t in chatbot.TOOLS}
    assert names == {"list_modules", "risk_list", "explain_student", "draft_letter",
                     "meeting_slots", "model_info"}


@pytest.mark.parametrize("name,args", [("explain_student", {}), ("no_such_tool", {}),
                                       ("risk_list", {"limit": "many"})])
def test_bad_tool_calls_return_errors(name, args):
    assert "error" in chatbot.call_tool(name, args, WEEK)
