# Changelog

## v0.2.5 (2026-03-22) [unreleased]
- fix: use plain string license field for setuptools compatibility
- docs: regenerate API reference [skip ci]
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.2.5 (2026-03-21)
- docs: replace pip install with uv add in README
- Fix benchmark: replace latent-factor DGP with pure Sarmanov DGP
- docs: regenerate API reference [skip ci]
- make torch optional: move to [neural] extra with lazy imports
- Add blog post link and community CTA to README
- fix: update license badge from BSD-3 to MIT
- docs: regenerate API reference [skip ci]
- fix: QA audit batch 5 — formula corrections and doc accuracy (v0.2.4)
- Add PyPI classifiers for financial/insurance audience
- Add Colab quickstart notebook and Open in Colab badge
- Update Performance section with post-review benchmark results
- docs: regenerate API reference [skip ci]
- fix: correct P0/P1 bugs from code review (v0.2.3)
- docs: regenerate API reference [skip ci]
- Add benchmark: Sarmanov copula joint freq-sev vs independence assumption
- pin statsmodels>=0.14.5 for scipy compat
- Fix ConditionalFreqSev.premium_correction() shape mismatch (v0.2.1)
- Add shields.io badge row (PyPI, Python, Tests, License)
- docs: add Databricks notebook link
- Add Related Libraries section to README
- fix: README code blocks — add synthetic data, fix API bug in to_html call
- fix: replace np.trapz with numpy 2.x compat call in dependent diagnostics

