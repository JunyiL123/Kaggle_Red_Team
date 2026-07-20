# Gated EXFIL ladder (1x → 2x → 3x plain)

Submission for the [Kaggle AI Agent Security](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks) public leaderboard.

## Strategy

1. Probe **all 6** short **1x** wordings (no early-stop)
2. Rank by **conservative raw points/sec** (`16 × SECRET_MARKER posts + 2`)
3. Screen **2x plain** dual; promote only if points/sec clearly beats 1x
4. If 2x wins, screen **3x plain**; promote only if it beats 2x
5. Farm the selected rung; **probation rollback** if live rate collapses
6. Cap returns by measured replay cost (`REPLAY_SAFE=0.94`)
7. Keep only trials that actually `http.post` **`SECRET_MARKER`** (plaintext; no reverse/hex/b64)

Reference competitor search pattern: `kaggle_outputs/tetsutani/` (we keep their gating, not their reversed second payload).

**A/B reversible variant:** [`attack_reversible.py`](attack_reversible.py) + [`kaggle_submit_reversible.ipynb`](kaggle_submit_reversible.ipynb) — same ladder, but 2x/3x challengers also send reverse / hex / base64 forms and count them at search time.

Archived Optimized code: [`attack_optimized.py`](attack_optimized.py).

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

Use [`kaggle_submit.ipynb`](kaggle_submit.ipynb): Save & Run All (Internet OFF + GPU) → Submit that Version. Run **2–3×** before changing code. Check Output for `attack_run_summary.txt` if present.
