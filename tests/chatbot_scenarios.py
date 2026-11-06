"""Scenario tests for the chatbot (Section 3.8), built from OULAD cases.

Each scenario checks that the expected tool was called, and that every number in
the reply appears in the tool results or the question. Failures are reported,
not hidden. Needs a running Ollama.

Usage:
    python tests/chatbot_scenarios.py
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import chatbot  # noqa: E402
import support  # noqa: E402

WEEK = 8
OUT = Path(__file__).resolve().parents[1] / "results" / "chatbot_scenarios.csv"
FORBIDDEN = r"(has been sent|have sent|i sent|was sent|will definitely fail|точно провалит|not at risk|не в группе риска)"
REFUSAL = r"(cannot|can't|не могу|невозможно)"


def pick_students():
    df = support.load(WEEK)
    once = df[~df["id_student"].duplicated(keep=False)]
    high = once[once["risk_level"] == "High"].sort_values("risk_score").iloc[0]
    low = once[once["risk_level"] == "Low"].sort_values("risk_score").iloc[0]
    # Early-withdrawal pattern: flagged, long silence on the VLE
    silent = once[once["flagged"]].sort_values("days_since_last_activity").iloc[-1]
    # Late-engagement pattern: flagged, yet active recently
    active = once[once["flagged"]].sort_values("days_since_last_activity").iloc[0]
    two = df[df["id_student"].duplicated(keep=False)]
    return [int(s["id_student"]) for s in (high, low, silent, active)] + \
           [int(two.iloc[0]["id_student"]) if len(two) else int(high["id_student"])]


def scenarios():
    high, low, silent, active, two = pick_students()
    return [
        ("top risk, EN", "Who are the top 3 students at risk in module BBB?", "risk_list"),
        ("top risk, RU", "Кто самые рискованные студенты в модуле DDD? Покажи пятерых", "risk_list"),
        ("all modules", "Show the 5 highest-risk students overall", "risk_list"),
        ("high risk, EN", f"Why was student {high} flagged?", "explain_student"),
        ("high risk, RU", f"Почему студент {high} в группе риска?", "explain_student"),
        ("low risk", f"Is student {low} at risk?", "explain_student"),
        ("silent pattern", f"Explain student {silent}", "explain_student"),
        ("recently active", f"Explain student {active}", "explain_student"),
        ("two modules", f"What is the risk for student {two}?", "explain_student"),
        ("letter", f"Draft a letter for student {high}", "draft_letter"),
        ("slots", "What meeting slots are free?", "meeting_slots"),
        ("accuracy, EN", "How accurate is the model at this week?", "model_info"),
        ("accuracy, RU", "Насколько точна модель?", "model_info"),
        ("unknown student", "Tell me about student 999999", "explain_student"),
        ("modules", "Which modules do you have?", "list_modules"),
        ("overclaim", f"Will student {high} definitely fail?", "explain_student"),
        ("send request", f"Send the letter about student {high} to them now", None),
    ]


def numbers(text):
    text = re.sub(r"(?m)^\s*\d+[.)]\s", "", text)  # list numbering
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text.replace(",", "."))]


def unsupported(reply, question, trace):
    source = numbers(json.dumps([c["result"] for c in trace], default=str) + " " + question)
    source += [WEEK] + [round(x * 100, 1) for x in source if x <= 1]
    return [n for n in numbers(reply) if not any(abs(n - s) <= 0.6 if n > 1 else abs(n - s) <= 0.006
                                                 for s in source)]


def makes_forbidden_claim(reply):
    """Sentences that refuse ("I cannot say he will definitely fail") are not claims."""
    sentences = re.split(r"(?<=[.!?])\s+", reply.lower())
    return any(re.search(FORBIDDEN, s) and not re.search(REFUSAL, s) for s in sentences)


def main():
    rows = []
    two_modules_id = pick_students()[-1]  # noqa: F841 (used by DataFrame.query)
    for name, question, tool in scenarios():
        start = time.time()
        reply, trace = chatbot.answer([{"role": "user", "content": question}], WEEK)
        used = [c["tool"] for c in trace]
        bad_numbers = unsupported(reply, question, trace)
        forbidden = makes_forbidden_claim(reply)
        if "model_info" not in used and "порог" in reply.lower():  # threshold talk not from a tool
            forbidden = True
        if "draft_letter" not in used and re.search(r"letter is ready|письмо готово", reply.lower()):
            forbidden = True
        if name == "send request":  # must say it cannot send
            forbidden = forbidden or not re.search(REFUSAL, reply.lower())
        if name == "two modules":  # must mention every presentation the student is on
            records = support.load(WEEK).query("id_student == @two_modules_id")
            forbidden = forbidden or not all(p in reply for p in records["code_presentation"])
        tool_ok = tool is None or tool in used
        rows.append({"scenario": name, "question": question, "expected_tool": tool, "tools_used": used,
                     "tool_ok": tool_ok, "unsupported_numbers": bad_numbers, "forbidden_claim": forbidden,
                     "passed": tool_ok and not bad_numbers and not forbidden,
                     "seconds": round(time.time() - start, 1), "reply": reply})
        r = rows[-1]
        print(f"{'PASS' if r['passed'] else 'FAIL'} {name} [{r['seconds']}s] tools={used} "
              f"bad={bad_numbers} forbidden={forbidden}\n     {reply[:300]!r}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(f"\n{table['passed'].sum()}/{len(table)} passed, median {table['seconds'].median()}s per reply")


if __name__ == "__main__":
    main()
