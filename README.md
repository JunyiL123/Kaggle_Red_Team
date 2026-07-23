# Three A/B attack variants

Submission for the [Kaggle AI Agent Security](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks) public leaderboard.

## Submit these three

| Notebook | Source | What it is |
|---|---|---|
| [`kaggle_submit_tetsutani.ipynb`](kaggle_submit_tetsutani.ipynb) | [`attack_tetsutani.py`](attack_tetsutani.py) | Near-copy of competitor v139 |
| [`kaggle_submit.ipynb`](kaggle_submit.ipynb) | [`attack.py`](attack.py) | **Ours, no Harmony** — raw/sec, screen-all duals, stack if 2+ hit |
| [`kaggle_submit_harmony.ipynb`](kaggle_submit_harmony.ipynb) | [`attack_harmony.py`](attack_harmony.py) | **Ours + Harmony** — inj-wrap duals, no 1.05 gate, optional stack_2 only |

Smoke A/B (not Submit): [`kaggle_debug_harmony_dual1.ipynb`](kaggle_debug_harmony_dual1.ipynb) vs [`kaggle_debug_harmony_dual2.ipynb`](kaggle_debug_harmony_dual2.ipynb) (`DUAL_SCREEN_REPS` 1 vs 2). Equal-N H2H: [`kaggle_debug_harmony_h2h.ipynb`](kaggle_debug_harmony_h2h.ipynb).

Older reference only: [`kaggle_submit_reversible.ipynb`](kaggle_submit_reversible.ipynb), [`kaggle_outputs/tetsutani/`](kaggle_outputs/tetsutani/).

## Setup / local checks

```bash
cd "/Users/junyiliu/Desktop/AI Safety Internship/Kaggle Attack Red Team"
source .venv/bin/activate
pip install -r requirements.txt
aicomp validate redteam attack.py
python demo_loop.py
python -m pytest tests/test_attack_upgrades.py -v
```

## Submit on Kaggle

For each notebook: Save & Run All (Internet OFF + GPU) → Submit that Version. Prefer **2–3×** per variant before changing code. Check Output for `attack_run_summary.txt` if present.
