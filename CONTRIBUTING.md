# Contributing to servefit

Issues and pull requests are welcome. A few ground rules:

- **The matrix is the product.** Any change to `servefit/engines.py` must keep
  every row's `check_date` current and cite a source. If you cannot verify a
  cell, say so in the PR - do not guess.
- **Honest limits over clever claims.** servefit never says "production ready".
  Keep the "Missing for production" lists truthful and the assumptions printed.
- **Zero dependencies.** The package must stay stdlib-only so it runs wherever
  Python 3.10+ exists. Tests may use `pytest` only.
- **Tests for every change.** Run `pytest -q` before opening a PR; CI runs
  3.10-3.13.
- **English everywhere** in user-facing text (README, docs, CLI output).
- **No synthetic-looking engagement.** Do not open issues/PRs to inflate
  activity.

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install pytest
pytest -q
python -m servefit --version
```

## Re-rendering the demo

```bash
python scripts/make-demo-gif.py   # rebuilds examples/demo.gif from real CLI output
```
