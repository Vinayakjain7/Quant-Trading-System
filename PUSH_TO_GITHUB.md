# How to update your GitHub repo

Run these from the repo root: `quant trading/`.

## If this folder is NOT yet a git repo

```bash
git init
git add .
git commit -m "feat: v2 — rigorous backtest, ML ranking layer, risk controls, tests & CI"
git branch -M main
git remote add origin https://github.com/Vinayakjain7/quant-trading.git
git push -u origin main
```

## If it IS already connected to your existing repo

```bash
git add .
git commit -m "feat: v2 — rigorous backtest, ML ranking layer, risk controls, tests & CI"
git push
```

The old code is preserved untouched in `old system/`, so your history is intact.

## Before you push — quick checklist

1. README badges/URLs are already set to `Vinayakjain7/quant-trading`. If you name the
   GitHub repo something other than `quant-trading`, update those references to match.
2. Make sure `venv/`, `data_cache/`, and `outputs/` are **not** committed — `.gitignore`
   already excludes them. Verify with `git status` before committing.
3. Run the test suite once: `pytest -q` — keep CI green so the badge shows passing.

## What actually earns stars (honest version)

Stars come from a repo being **useful and trustworthy**, not from the code alone:

- A clear README with a real "why this is different" angle — yours now has the
  "we don't cheat the backtest" table, which is genuinely uncommon in hobby repos.
- Green CI badge + tests signal the project works.
- A short demo: commit one sample results screenshot or a `sample_backtest.png`
  (run `python -m quant.cli backtest`, screenshot the metrics) and embed it near the top.
- Add good GitHub **topics**: `quant`, `algorithmic-trading`, `backtesting`,
  `python`, `machine-learning`, `nse`, `yfinance`.
- Write a short post (Reddit r/algotrading, LinkedIn, X) explaining the look-ahead-bias
  angle and link the repo. Distribution matters more than code for stars.
```
