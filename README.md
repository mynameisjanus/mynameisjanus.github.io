# Building Quantitative Trading Systems

Source for [mynameisjanus.github.io](https://mynameisjanus.github.io/) — a course on how professional quantitative trading systems are actually built: market structure, statistics, strategy research, a hand-built backtesting engine, live trading infrastructure, and running a systematic trading business.

Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve          # live-reload preview at http://127.0.0.1:8000
mkdocs build --strict # the same gate CI uses
```

## Deployment

Every push to `main` triggers `.github/workflows/ci.yml`, which runs `mkdocs build --strict` and deploys the built site to GitHub Pages.

One-time setup (already done, documented for reference): repository **Settings → Pages → Source → "GitHub Actions"**. This bypasses GitHub's native Jekyll build entirely.

## Analytics

The site currently ships without analytics. To add Google Analytics 4 later, create a GA4 property and add to `mkdocs.yml`:

```yaml
extra:
  analytics:
    provider: google
    property: G-XXXXXXXXXX
```

## Repository notes

- `docs/appendix/` is an 18-part mathematics, probability, and statistics reference organized as `part-NN-*/` directories. Much of its content was migrated from the previous Jekyll site ("The Science of Data"); the one-off converter is kept at `scripts/migrate_math_pages.py` for provenance. Pages marked "Draft" are placeholders.
- Course content is © Janus B. Advincula, all rights reserved (see `LICENSE`).
