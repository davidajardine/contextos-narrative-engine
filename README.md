# ContextOS Narrative Strategy Engine 🧠📊

This repository contains experimental Python code for backtesting, analyzing, and refining a **narrative-based intraday trading strategy**. It represents a unique approach to interpreting market behavior not through strict indicators, but through **structural storytelling, session mechanics, and pattern recognition**.

## 🧭 Strategy Philosophy

Unlike traditional quant models, this system assumes **market structure behaves like a narrative**:
- Sessions have **roles** (e.g., Brinks as the decision-maker).
- Price moves in **acts** (e.g., Raid → Displacement → Retrace → True Move).
- Reversals and continuations are treated as **characters reappearing with intent**.
- Volume and wicks are **emotional beats**.

This is built with the goal of modeling the mindset of a disciplined human trader using a **logical narrative flow**, and is most aligned with setups like *Wormsign*, *Power Run*, *Shakeout*, and other session-based MM behavior.

## 📂 Structure

Included scripts are mostly self-contained and reflect the iterative growth of the system over time:

| File | Description |
|------|-------------|
| `simple_narrative.py` | Early attempt at defining narrative session flow. |
| `simple_narrative_3h.py` | Adds 3-hour structural timeframes. |
| `simple_narrative_3h_offset.py` | Adjusts for session alignment based on offset logic. |
| `final_simple_narrative_3h_offset*.py` | Final iterations with debug and volume/event logic. |
| `new_aggregated_set.py` | [Optional] Blends various setup types or aggregates decision weights. |

All files are written in Python and are **exploratory in nature**. Results are not optimized, and many scripts will require environmental setup (such as pre-loaded DataFrames or CSVs).

## ⚠️ Disclaimer

> This code is **experimental**, **unoptimized**, and likely to contain logic inconsistencies, unhandled edge cases, or performance issues. Use for research purposes only.

---

## 🔧 Getting Started

```bash
# Create a new virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required libraries
pip install pandas numpy matplotlib

The code assumes you have access to structured session-level candle data, preferably in a CSV format with labeled sessions (Asia, London, NY).

📈 Vision
This project is part of a larger vision to develop a modular AI-ready strategy engine with multi-model session interpreters, capable of being trained on real data to identify setups and adaptively classify trade scenarios.

📜 License
MIT License — see LICENSE.md for details.
