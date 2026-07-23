# AGENTS.md

## Cursor Cloud specific instructions

This is an offline Python (3.12) research/competition codebase for the Kaggle
"AI Agent Security — Multi-Step Tool Attacks" red-team challenge. It has **no
web servers, databases, or long-running services** — everything is one-shot CLI
/ library / notebook work run against the simulated sandbox provided by the
`aicomp-sdk` package. There are no ports to bind.

Key entry points and how to run them (dependencies are already installed by the
update script; see `requirements.txt` — the single dep `aicomp-sdk` pulls in a
large tree incl. torch/transformers/CUDA wheels):

- Validate a submission: `aicomp validate redteam attack.py`
- Run the test suite: `python -m pytest tests/test_attack_upgrades.py -v`
- Local end-to-end smoke run: `python demo_loop.py` (~20s budgeted attack loop
  against the deterministic sandbox agent).

Non-obvious gotchas:

- The `aicomp` console script installs to `~/.local/bin`, which is **not always
  on `PATH`**. If `aicomp` is not found, run `export PATH="$HOME/.local/bin:$PATH"`
  or invoke as `python -m aicomp ...`.
- Packages install to the user/site path (no virtualenv); Ubuntu's
  `python3-venv` is not installed, so `python3 -m venv` fails. Use the system
  `python3`/`pip3` directly (the update script does this).
- `demo_loop.py` and the tests use the built-in `deterministic` agent, which
  never posts the real `SECRET_MARKER`. So `returned=0` / "No SECRET_MARKER wins
  kept" is the **expected, correct** local outcome — it is not a failure. Real
  scoring only happens on Kaggle with model-backed agents (`gpt_oss`/`gemma`).
- Running `demo_loop.py` writes `attack_run_summary.txt` in the repo root; it is
  git-ignored.
- The three submission variants are `attack.py` (base), `attack_harmony.py`,
  and `attack_tetsutani.py`, each paired with a `kaggle_submit_*.ipynb`. Kaggle
  notebooks are meant to be run with Internet OFF + GPU.
