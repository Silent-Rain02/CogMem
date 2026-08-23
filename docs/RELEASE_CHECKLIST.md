# Public-release checklist

This repository is deliberately in pre-release form. Before changing its visibility, complete the following checks.

- [ ] Confirm the paper's publication status and add the final paper URL and BibTeX to the README.
- [ ] Verify repository title, authors, affiliations, and maintainer contact details.
- [ ] Keep LoCoMo and LongMemEval out of the release branch unless their licenses explicitly permit redistribution; validate every image and result artifact independently.
- [ ] Scan the complete history and working tree for credentials, access tokens, private URLs, raw conversations, and user-identifying metadata.
- [ ] Run `uv sync --extra dev`, `uv run pytest`, and the documented one-sample evaluation smoke tests from a clean checkout.
- [ ] Re-run the final benchmark configuration, archive commands and output files privately, and verify every table in the README against the camera-ready record.
- [ ] Add repository topics, issue templates, a maintainer response policy, and a release tag.
- [ ] Review the MIT license holder/year and update only with the rights holder's approval.
