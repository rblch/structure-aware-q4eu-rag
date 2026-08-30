# Structure-Aware RAG on Q4EU

Implementation of an experiment on structure-aware retrieval and context
assembly for legal question answering over Q4EU from DiscoLQA. It compares
fixed-size, semantic, and legal-structure-aware chunking for retrieval, then
evaluates whether structural enrichment improves generated-answer faithfulness.

## Setup

Model calls are routed through OpenRouter.

```bash
cp .env.example .env   # fill in OPENROUTER_API_KEY; .env is gitignored
```

## Running

The raw DiscoLQA sources are committed under `data/raw/`, so a run starts at
`parse`. Stages must run in order:

```bash
python3 -B src/pipeline.py parse                # corpus parsing and audits
python3 -B src/pipeline.py chunk --include-semantic
python3 -B src/pipeline.py index                # embeddings + FAISS indices
python3 -B src/pipeline.py retrieve             # RQ1 retrieval evaluation
python3 -B src/pipeline.py assemble             # enrichment contexts
python3 -B src/pipeline.py generate             # answer generation
python3 -B src/pipeline.py evaluate-generation  # faithfulness + correctness
python3 -B src/pipeline.py analyze              # summary, tables, figures
python3 -B src/pipeline.py audit                # manual judge-audit packet
```

Faithfulness is judged by Claude Haiku 4.5 and correctness by Gemini 3.5
Flash. The two outcomes use independent evaluator configurations.

Every stage takes `--help`. Omitting `--include-semantic` from `chunk` skips the
only API-dependent part of that stage. To re-fetch the pinned upstream sources
instead of using the committed copy:

```bash
python3 src/data_import/discolqa_fetch.py \
  --commit f8614b7d75d4054fb6b02119cde92492172fd6b6
```

The post-experiment RQ3 extension reuses the hierarchical `no_enrichment`
answers and generates only the 72 fixed-size and 72 semantic comparators:

```bash
python3 -B src/pipeline.py assemble-rq3
python3 -B src/pipeline.py generate-rq3 \
  --pilot-query-count 5 --output-dir data/generation_rq3/pilot
python3 -B src/pipeline.py generate-rq3
python3 -B src/pipeline.py analyze-rq3
```

The RQ3 analysis uses recorded token, cost, and latency data and does not call
the answer judges. Its outputs are isolated under
`data/results/rq3_cross_chunking/`.

The `audit` stage writes `data/audit/manual_judge_audit_sample.csv`; complete its
`manual_*`, `reviewer_id`, and `reviewed_at_utc` columns before citing the audit
as evidence for LLM-judge validity.

For a local, blinded review UI, install the optional dependency and launch:

```bash
uv sync --extra audit-ui
uv run --extra audit-ui streamlit run src/audit_ui.py
```

The first step records independent faithfulness and correctness assessments
before revealing automatic scores or rationales. Reviews are saved separately
under `data/audit/reviews/`, and the completed merged packet can be downloaded
from the UI. The generated sample CSV is not modified.

## Tests

```bash
uv run --extra dev python -m pytest
```

Some tests read pipeline outputs and only pass once the stages that produce them
have run.

## Reproducibility

- The upstream DiscoLQA source is pinned by commit in `config/config.yaml`.
- Raw imports record SHA-256 hashes in `data/raw/discolqa/source_metadata.json`.
- Confirmatory configurations are declared in `config/config.yaml` before runs.

## Source Dataset

- DiscoLQA repository: https://github.com/Francesco-Sovrano/DiscoLQA
- DiscoLQA paper: https://doi.org/10.1007/s10506-023-09387-2
