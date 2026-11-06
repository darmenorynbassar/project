"""Advisor chatbot: local Qwen (through Ollama) calling the support-layer tools.

The language model only chooses tools and phrases their results. It has no
tool that approves, sends or records anything, and all data stays on this
machine.

Usage (needs a running Ollama with the model pulled):
    python src/chatbot.py "Who is most at risk in module BBB?"
"""
import json
import sys

import ollama

import support

MODEL = "qwen3:4b-instruct"
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """You are an assistant for academic advisors in an early-warning system.
Current cut-off: week {week}.

Rules:
- Use a tool for every question about students, risk, letters, meeting slots or model quality.
- Call a tool even if the question asks for a prediction about one student, then report what it returns.
- Use only numbers and facts returned by tools. Never guess or invent them.
- Report each factor with its value and its effect ("raises risk" or "lowers risk"), and add no
  other explanation such as thresholds or comparisons that the tool did not give.
- If a student has several records, report every module.
- If a tool finds nothing, repeat its note without changing the meaning.
- Present risk as decision support, not as a fact about the student. Never tell the advisor
  a student will fail.
- "Low" means not flagged by the model, not safe. Never say a student is not at risk.
- You cannot send, approve or record anything. If asked to, say clearly that you cannot, and
  that the advisor reviews and saves decisions in the Student review tab.
- Only after calling draft_letter, say the letter is ready for review in the Student review tab,
  and do not repeat it.
- Answer briefly, in the language of the advisor's last message.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_modules",
        "description": "List module presentations with the number of students and flagged students.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "risk_list",
        "description": "Flagged at-risk students, highest risk first.",
        "parameters": {"type": "object", "properties": {
            "module": {"type": "string", "description": "Module code, e.g. BBB. Omit for all modules."},
            "limit": {"type": "integer", "description": "How many students to return. Default 10."}}}}},
    {"type": "function", "function": {
        "name": "explain_student",
        "description": "Risk level and top factors for one student.",
        "parameters": {"type": "object", "required": ["student_id"], "properties": {
            "student_id": {"type": "integer"},
            "module": {"type": "string", "description": "Module code, only if the student has several."}}}}},
    {"type": "function", "function": {
        "name": "draft_letter",
        "description": "Draft an advisor letter about one student for human review.",
        "parameters": {"type": "object", "required": ["student_id"], "properties": {
            "student_id": {"type": "integer"},
            "module": {"type": "string", "description": "Module code, only if the student has several."}}}}},
    {"type": "function", "function": {
        "name": "meeting_slots",
        "description": "Free meeting slots in the mock calendar.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "model_info",
        "description": "How accurate the model is at this week, and what the risk score means.",
        "parameters": {"type": "object", "properties": {}}}},
]


def call_tool(name, args, week):
    try:
        if name == "list_modules":
            return support.modules(week)
        if name == "risk_list":
            return support.risk_list(week, args.get("module"), limit=int(args.get("limit") or 10))
        if name == "explain_student":
            return support.explain(args["student_id"], week, args.get("module"))
        if name == "draft_letter":
            return support.draft_letter(args["student_id"], week, args.get("module"))
        if name == "meeting_slots":
            return support.meeting_slots()
        if name == "model_info":
            return support.model_info(week)
        return {"error": f"unknown tool {name}"}
    except (KeyError, TypeError, ValueError) as e:
        return {"error": f"bad arguments for {name}: {e}"}


def answer(history, week):
    """history: [{'role', 'content'}, ...]. Returns the reply and the tool calls made."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(week=week)}, *history]
    trace = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS, think=False,
                               options={"temperature": 0.1, "num_ctx": 4096})
        message = response.message
        if not message.tool_calls:
            return message.content.strip(), trace
        messages.append(message)
        for call in message.tool_calls:
            name, args = call.function.name, dict(call.function.arguments)
            result = call_tool(name, args, week)
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, default=str)})
    return "I could not finish this request. Please try rephrasing it.", trace


if __name__ == "__main__":
    reply, calls = answer([{"role": "user", "content": " ".join(sys.argv[1:])}], week=8)
    print(reply)
    print("\nTools used:", [c["tool"] for c in calls])
