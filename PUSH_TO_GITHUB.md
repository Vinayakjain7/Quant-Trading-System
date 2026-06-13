# How to update your GitHub repo

Your repo already exists at **https://github.com/Vinayakjain7/Quant-Trading-System**
(it has the basic version), but this local folder isn't a git repo yet. Run everything
below from the repo root: `quant trading/`.

## Recommended: replace the old version with v2 (clean)

This overwrites the repo's contents with your new, polished version. The old basic
code is preserved locally in `old system/`, so nothing is lost.

```bash
git init
git add .
git commit -m "feat: v2 — rigorous backtest, regime filter, ML ranking, tuned & pruned"
git branch -M main
git remote add origin https://github.com/Vinayakjain7/Quant-Trading-System.git
git push -u origin main --force
```

The `--force` is needed because the local and remote histories are unrelated
(the remote has the old version with no shared commits). It's safe here — it's your
own repo and you're intentionally replacing the old code.

## Alternative: keep the old commit history

If you'd rather merge instead of overwrite:

```bash
git init
git add .
git commit -m "feat: v2 — rigorous backtest, regime filter, ML ranking, tuned & pruned"
git branch -M main
git remote add origin https://github.com/Vinayakjain7/Quant-Trading-System.git
git pull origin main --allow-unrelated-histories   # resolve any conflicts, then:
git push -u origin main
```

## Before you push — quick checklist

1. README badges/URLs are already set to `Vinayakjain7/Quant-Trading-System` — no edits needed.
2. Make sure `venv/`, `data_cache/`, and `outputs/` are **not** committed — `.gitignore`
   already excludes them. Verify with `git status` before committing.
3. Run the test suite once: `pytest -q` — keep CI green so the badge shows passing.

## What actually earns stars (honest version)

Stars come from a repo being **useful and trustworthy**, not from the code alone:

- A clear README with a real "why this is different" angle — yours now has the
  "we don't cheat the backtest" table, which is genuinely uncommon in hobby repos.
- Green CI badge + tests signal the project works.
- A short demo: commit one sample results screenshot or a `sample_backtest.png`
  is already embedded near the top (regenerate with `python scripts/make_results_chart.py`).
- Add good GitHub **topics**: `quant`, `algorithmic-trading`, `backtesting`,
  `python`, `machine-learning`, `nse`, `yfinance`.
- Write a short post (Reddit r/algotrading, LinkedIn, X) explaining the look-ahead-bias
  angle and link the repo. Distribution matters more than code for stars.
```
