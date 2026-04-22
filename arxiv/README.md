# arxiv/

LaTeX source for the arXiv preprint on clustered retrieval-induced forgetting.

The preprint is scoped narrowly to the novel retrieval mechanism (clustered RIF + rank-gap suppression). The broader research journey (biology-inspired failures, LLM enrichment, shipped tool) lives in `../writeup/` and is intentionally out of scope here.

## Files

- `paper.tex` — single-file LaTeX source, `article` class, inline bibliography
- `paper.pdf` — generated, gitignored

## Build

`tectonic` is the recommended engine. Single binary, no sudo, downloads TeX packages on demand.

```
brew install tectonic
tectonic arxiv/paper.tex
```

Produces `arxiv/paper.pdf`. Cross-references resolve on the first pass.

### Alternative: pdflatex (BasicTeX or MacTeX)

If you need the actual `pdflatex` binary (for example to match a venue submission system that rejects tectonic output):

```
brew install --cask basictex          # ~100 MB, requires sudo for the .pkg installer
eval "$(/usr/libexec/path_helper)"    # picks up /Library/TeX/texbin
cd arxiv && pdflatex paper.tex && pdflatex paper.tex   # twice for cross-refs
```

Neither BasicTeX nor tectonic touches `pyproject.toml`, so the `lethe-memory` PyPI package is unaffected.

## arXiv submission bundle

arXiv wants source, not a PDF. When ready:

```
cd arxiv
tar czf lethe-arxiv.tar.gz paper.tex
# upload lethe-arxiv.tar.gz at https://arxiv.org/submit
```

If figures or a separate `.bib` get added later, include them in the tar. arXiv runs pdflatex on the source, so keep the LaTeX compatible with a plain pdflatex build (tectonic-specific commands are fine because the paper doesn't use any).

## Before submitting

Inline `[TODO]` markers in `paper.tex` flag the remaining work:

1. **Author metadata.** `\author{...}` block has placeholder email. Fill in before submission.
2. **Second dataset.** The evaluation section and limitations reference a planned replication on a second benchmark (BEIR subset, HotpotQA, or similar). Running clustered RIF on one more dataset is the single most important remaining experiment before submission. Results go in a new subsection in §5.
3. **Bootstrap confidence intervals.** The main results table reports point estimates. 10k-resample bootstrap CIs over the per-query NDCG differences from each system would strengthen the claim. Per-query NDCG arrays are saved by the benchmark runs under `benchmarks/results/`.
4. **Citation verification.** The query-clustering citation in §6 is flagged explicitly. Also double-check author lists and venue for the MemoryBank (Zhong et al.) and LongMemEval (Wu et al.) entries.
5. **Compile and eyeball.** `tectonic paper.tex`, open the PDF, check that all tables render, all cross-references resolve (no `??` in the text), and the bibliography renders correctly.

## Category

Intended arXiv categories: **cs.IR** (primary) with **cs.CL** cross-list. If targeting a venue later (SIGIR, EMNLP, ACL), swap the `\documentclass` line; the rest of the content carries over with minor formatting.

## Version control

- `paper.tex` is tracked in git
- `paper.pdf` is gitignored (generated output)
- If a stable version is submitted to arXiv, tag it: `git tag arxiv-v1`, `arxiv-v2` for revisions

The first-submission workflow: compile locally, review, commit the final `.tex`, tag, then upload the source to arXiv.
