"""Gradio demo for Open Alternative to Jev: one context, several typed questions, one forward pass.

Runs on a Hugging Face Space (ZeroGPU when available) or locally:

    SO1_DEMO_MODEL=Qwen/Qwen2.5-0.5B-Instruct python demo/app.py
"""
import html
import os
import time

import gradio as gr
import torch

from so1 import Choice, Decider
from so1.backends.hf import HFBackend

try:  # ZeroGPU on Hugging Face Spaces
    import spaces
    GPU = spaces.GPU
except Exception:  # local or CPU Space
    def GPU(fn=None, **_):
        return fn if fn is not None else (lambda f: f)

CUDA = torch.cuda.is_available()
MODEL_ID = os.environ.get("SO1_DEMO_MODEL", "Qwen/Qwen3.5-4B" if CUDA else "Qwen/Qwen2.5-1.5B-Instruct")
MAX_QUESTIONS = 8

backend = HFBackend.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16 if CUDA else torch.float32,
    device_map="cuda" if CUDA else None,
    batch_size=1,
)
decider = Decider(backend)


def parse_options(text: str) -> list[str]:
    return [o.strip() for o in text.replace("\n", ",").split(",") if o.strip()]


@GPU(duration=60)
def decide(state, mode, temperature, *fields):
    questions = []
    for i in range(MAX_QUESTIONS):
        q, opts = fields[2 * i], fields[2 * i + 1]
        if q and q.strip():
            options = parse_options(opts or "")
            if len(options) < 2:
                raise gr.Error(f"Question {i + 1} needs at least two options, separated by commas.")
            if len(options) > 26:
                raise gr.Error(f"Question {i + 1} has more than 26 options.")
            questions.append(Choice(q.strip(), options, name=f"q{i + 1}"))
    if not questions:
        raise gr.Error("Add at least one question.")
    decider.temperature = float(temperature)
    t0 = time.perf_counter()
    decisions = decider.decide(state.strip() or None, questions, mode=mode)
    ms = (time.perf_counter() - t0) * 1000
    rows = []
    for d in decisions:
        bars = "".join(
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
            f'<span style="width:140px;text-align:right;font-size:13px">{html.escape(o)}</span>'
            f'<div style="flex:1;background:#eee;height:14px;border-radius:3px"><div style="width:{p * 100:.1f}%;height:14px;'
            f'background:{"#2a78d6" if o == d.choice else "#b8b6b0"};border-radius:3px"></div></div>'
            f'<span style="width:52px;font-size:13px">{p * 100:.1f}%</span></div>'
            for o, p in zip(d.question.options, d.probabilities))
        rows.append(
            f'<div style="padding:10px 0;border-bottom:1px solid #ddd">'
            f'<div style="font-size:14px;color:#555">{html.escape(d.question.question)}</div>'
            f'<div style="font-size:18px;font-weight:600;margin:4px 0">{html.escape(d.choice)} '
            f'<span style="font-weight:400;color:#555;font-size:14px">confidence {d.confidence:.2f}</span></div>{bars}</div>')
    summary = (f"{len(decisions)} decision{'s' if len(decisions) > 1 else ''} in **one forward pass**, "
               f"{ms:.0f} ms including tokenization, model `{MODEL_ID}`, mode `{mode}`. "
               "Probabilities are a softmax over the option letters; raw confidence is usually too high, "
               "see the calibration section of the README.")
    return "".join(rows), summary, [d.as_dict() for d in decisions]


EXAMPLE_STATE = ("Ticket #8813 from a Pro-plan customer: 'Since yesterday's release the export button does nothing. "
                 "We have a board meeting Monday and need the PDF. Please fix or tell me a workaround.'")
EXAMPLE_QUESTIONS = [
    ("Ticket category", "bug, feature request, billing, question"),
    ("Urgency", "low, medium, high"),
    ("Should this be escalated to an engineer?", "yes, no"),
    ("Customer sentiment", "angry, worried, neutral, happy"),
]

with gr.Blocks(title="Open Alternative to Jev") as demo:
    gr.Markdown(
        "# Open Alternative to Jev\n"
        "Typed decisions from an open-weights LLM in **one forward pass**. Give it a context and any number of "
        "questions with fixed options; it never writes text, it only ranks your options and returns probabilities. "
        f"Model: `{MODEL_ID}`. Code and benchmarks: "
        "[github.com/ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev).")
    state = gr.Textbox(label="Context (the state: a ticket, an email, a log line, a paragraph)", lines=5, value=EXAMPLE_STATE)
    rows, q_boxes, o_boxes = [], [], []
    for i in range(MAX_QUESTIONS):
        with gr.Row(visible=i < len(EXAMPLE_QUESTIONS)) as row:
            q = gr.Textbox(label=f"Question {i + 1}", value=EXAMPLE_QUESTIONS[i][0] if i < len(EXAMPLE_QUESTIONS) else "", scale=3)
            o = gr.Textbox(label="Possible answers (comma-separated)", value=EXAMPLE_QUESTIONS[i][1] if i < len(EXAMPLE_QUESTIONS) else "", scale=3)
        rows.append(row)
        q_boxes.append(q)
        o_boxes.append(o)
    n_visible = gr.State(len(EXAMPLE_QUESTIONS))
    with gr.Row():
        add_btn = gr.Button("+ Add question")
        remove_btn = gr.Button("- Remove last question")
    with gr.Row():
        mode = gr.Radio(["packed", "separate"], value="packed", label="Mode",
                        info="packed: all questions in one sequence (fastest). separate: one sequence per question, no interference.")
        temperature = gr.Slider(0.5, 3.0, value=1.0, step=0.05, label="Temperature (calibration; 1.0 = raw)")
    run = gr.Button("Decide", variant="primary")
    out_html = gr.HTML()
    out_md = gr.Markdown()
    out_json = gr.JSON(label="Raw decisions", open=False)

    def add(n):
        n = min(n + 1, MAX_QUESTIONS)
        return [n] + [gr.Row(visible=i < n) for i in range(MAX_QUESTIONS)]

    def remove(n):
        n = max(n - 1, 1)
        return [n] + [gr.Row(visible=i < n) for i in range(MAX_QUESTIONS)]

    add_btn.click(add, n_visible, [n_visible] + rows)
    remove_btn.click(remove, n_visible, [n_visible] + rows)
    fields = [c for pair in zip(q_boxes, o_boxes) for c in pair]
    run.click(decide, [state, mode, temperature] + fields, [out_html, out_md, out_json])

if __name__ == "__main__":
    demo.launch()
