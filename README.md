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


To re-fetch the pinned upstream sources instead of using the committed copy:

```bash
python3 src/data_import/discolqa_fetch.py \
  --commit f8614b7d75d4054fb6b02119cde92492172fd6b6
```

## Reproducibility

- The upstream DiscoLQA source is pinned by commit in `config/config.yaml`.
- Raw imports record SHA-256 hashes in `data/raw/discolqa/source_metadata.json`.
- Confirmatory configurations are declared in `config/config.yaml` before runs.

## Source Dataset

- DiscoLQA repository: https://github.com/Francesco-Sovrano/DiscoLQA
- DiscoLQA paper: https://doi.org/10.1007/s10506-023-09387-2
