---
title: Open Alternative to Jev
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Typed decisions from an open LLM in one forward pass
---

# Open Alternative to Jev

Give it a context and any number of questions with fixed options. The model never writes text: it reads the
next-token distribution at one position per question, in a single forward pass, and returns the option with
its probability.

Model: Qwen3.5-4B on GPU (a smaller Qwen on CPU). Library, benchmarks on Qwen3.6-27B and the honest
correction of our first speed claim: [github.com/ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev).

Set the Space variable `SO1_DEMO_MODEL` to run another checkpoint.
