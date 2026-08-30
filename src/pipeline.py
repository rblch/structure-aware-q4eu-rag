from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from data_import.discolqa_fetch import assert_snapshot_matches_config, fetch_discolqa
from data_import.q4eu_parser import import_q4eu
from data_import.xref_eligibility import write_structural_eligibility_outputs
from chunking.chunk_corpus import write_chunk_outputs
from enrichment.context_assembly import write_context_assembly_outputs
from enrichment.rq3_context_assembly import write_rq3_context_assembly_outputs
from evaluation.generated_answer_evaluation import (
    parse_csv as parse_evaluation_csv,
    write_generated_answer_evaluation_outputs,
)
from evaluation.manual_audit import write_manual_audit_outputs
from generation.answer_generation import parse_csv, write_generation_outputs
from parsing.corpus_parser import write_corpus_outputs
from parsing.xref_extractor import write_xref_outputs
from analysis.reporting import write_stage8_outputs
from analysis.rq3_cross_chunking import write_rq3_cross_chunking_outputs
from retrieval.evaluate_retrieval import write_retrieval_outputs
from retrieval.index_corpus import write_index_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Q4EU RAG experiment stages.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Run Stage 0 import.")
    fetch_parser.add_argument("--commit", required=True)
    fetch_parser.add_argument("--repo", default="Francesco-Sovrano/DiscoLQA")
    fetch_parser.add_argument("--branch", default="main")
    fetch_parser.add_argument("--output-dir", default="data/raw/discolqa")

    parse_parser = subparsers.add_parser("parse", help="Run Stage 1 parsing and audits.")
    parse_parser.add_argument("--config-path", default="config/config.yaml")
    parse_parser.add_argument(
        "--evaluate-path",
        default="data/raw/discolqa/evaluate.py",
    )
    parse_parser.add_argument(
        "--source-metadata-path",
        default="data/raw/discolqa/source_metadata.json",
    )
    parse_parser.add_argument(
        "--raw-documents-dir",
        default="data/raw/discolqa/documents",
    )
    parse_parser.add_argument(
        "--queries-path",
        default="data/dataset/q4eu_queries.json",
    )
    parse_parser.add_argument(
        "--import-report-path",
        default="data/dataset/q4eu_import_report.json",
    )
    parse_parser.add_argument(
        "--legal-units-path",
        default="data/parsed/legal_units.json",
    )
    parse_parser.add_argument(
        "--canonical-texts-path",
        default="data/parsed/canonical_texts.json",
    )
    parse_parser.add_argument(
        "--validation-report-path",
        default="data/parsed/validation_report.json",
    )
    parse_parser.add_argument(
        "--xref-graph-path",
        default="data/parsed/xref_graph.json",
    )
    parse_parser.add_argument(
        "--xref-eligibility-report-path",
        default="data/dataset/xref_eligibility_report.json",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Run Stage 2 chunk generation.")
    chunk_parser.add_argument("--config-path", default="config/config.yaml")
    chunk_parser.add_argument(
        "--legal-units-path",
        default="data/parsed/legal_units.json",
    )
    chunk_parser.add_argument(
        "--canonical-texts-path",
        default="data/parsed/canonical_texts.json",
    )
    chunk_parser.add_argument("--output-dir", default="data/chunks")
    chunk_parser.add_argument(
        "--include-semantic",
        action="store_true",
        help="Generate semantic chunks using OpenRouter embeddings.",
    )
    chunk_parser.add_argument(
        "--characteristics-path",
        default="data/chunks/chunk_characteristics.json",
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Run Stage 3 embedding and indexing.",
    )
    index_parser.add_argument("--config-path", default="config/config.yaml")
    index_parser.add_argument("--chunks-dir", default="data/chunks")
    index_parser.add_argument("--embeddings-dir", default="data/embeddings")
    index_parser.add_argument("--indices-dir", default="data/indices")
    index_parser.add_argument(
        "--characteristics-path",
        default="data/indices/index_characteristics.json",
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Run Stage 4 retrieval evaluation.",
    )
    retrieve_parser.add_argument("--config-path", default="config/config.yaml")
    retrieve_parser.add_argument(
        "--queries-path",
        default="data/dataset/q4eu_queries.json",
    )
    retrieve_parser.add_argument("--chunks-dir", default="data/chunks")
    retrieve_parser.add_argument("--embeddings-dir", default="data/embeddings")
    retrieve_parser.add_argument("--indices-dir", default="data/indices")
    retrieve_parser.add_argument(
        "--retrieval-results-path",
        default="data/retrieval/retrieval_results.json",
    )
    retrieve_parser.add_argument(
        "--retrieval-metrics-path",
        default="data/evaluation/retrieval_metrics.json",
    )

    assemble_parser = subparsers.add_parser(
        "assemble",
        help="Run Stage 5 enrichment context assembly.",
    )
    assemble_parser.add_argument("--config-path", default="config/config.yaml")
    assemble_parser.add_argument(
        "--legal-units-path",
        default="data/parsed/legal_units.json",
    )
    assemble_parser.add_argument(
        "--queries-path",
        default="data/dataset/q4eu_queries.json",
    )
    assemble_parser.add_argument("--chunks-dir", default="data/chunks")
    assemble_parser.add_argument("--embeddings-dir", default="data/embeddings")
    assemble_parser.add_argument("--indices-dir", default="data/indices")
    assemble_parser.add_argument(
        "--xref-graph-path",
        default="data/parsed/xref_graph.json",
    )
    assemble_parser.add_argument(
        "--retrieval-results-path",
        default="data/retrieval/retrieval_results.json",
    )
    assemble_parser.add_argument(
        "--context-records-path",
        default="data/evaluation/context_assembly_records.json",
    )
    assemble_parser.add_argument(
        "--coverage-metrics-path",
        default="data/evaluation/context_coverage_metrics.json",
    )
    assemble_parser.add_argument(
        "--xref-eligibility-report-path",
        default="data/dataset/xref_eligibility_report.json",
    )

    assemble_rq3_parser = subparsers.add_parser(
        "assemble-rq3",
        help="Assemble fixed-size and semantic contexts for the RQ3 extension.",
    )
    assemble_rq3_parser.add_argument("--config-path", default="config/config.yaml")
    assemble_rq3_parser.add_argument(
        "--queries-path", default="data/dataset/q4eu_queries.json"
    )
    assemble_rq3_parser.add_argument("--chunks-dir", default="data/chunks")
    assemble_rq3_parser.add_argument(
        "--retrieval-results-path",
        default="data/retrieval/retrieval_results.json",
    )
    assemble_rq3_parser.add_argument(
        "--context-records-path",
        default="data/evaluation/rq3_context_assembly_records.json",
    )
    assemble_rq3_parser.add_argument(
        "--context-metrics-path",
        default="data/evaluation/rq3_context_metrics.json",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run Stage 6 answer generation.",
    )
    generate_parser.add_argument("--config-path", default="config/config.yaml")
    generate_parser.add_argument(
        "--context-records-path",
        default="data/evaluation/context_assembly_records.json",
    )
    generate_parser.add_argument(
        "--coverage-metrics-path",
        default="data/evaluation/context_coverage_metrics.json",
    )
    generate_parser.add_argument("--output-dir", default="data/generation")
    generate_parser.add_argument(
        "--prompt-path",
        default="config/prompts/answer_generation.txt",
    )
    generate_parser.add_argument("--condition-ids")
    generate_parser.add_argument("--query-ids")
    generate_parser.add_argument("--pilot-query-count", type=int)
    generate_parser.add_argument("--overwrite", action="store_true")

    generate_rq3_parser = subparsers.add_parser(
        "generate-rq3",
        help="Generate the 144 fixed-size and semantic answers for RQ3.",
    )
    generate_rq3_parser.add_argument("--config-path", default="config/config.yaml")
    generate_rq3_parser.add_argument(
        "--context-records-path",
        default="data/evaluation/rq3_context_assembly_records.json",
    )
    generate_rq3_parser.add_argument(
        "--context-metrics-path",
        default="data/evaluation/rq3_context_metrics.json",
    )
    generate_rq3_parser.add_argument("--output-dir", default="data/generation_rq3")
    generate_rq3_parser.add_argument(
        "--prompt-path", default="config/prompts/answer_generation.txt"
    )
    generate_rq3_parser.add_argument("--query-ids")
    generate_rq3_parser.add_argument("--pilot-query-count", type=int)
    generate_rq3_parser.add_argument("--overwrite", action="store_true")

    evaluate_generation_parser = subparsers.add_parser(
        "evaluate-generation",
        help="Run Stage 7 generated-answer evaluation.",
    )
    evaluate_generation_parser.add_argument(
        "--config-path",
        default="config/config.yaml",
    )
    evaluate_generation_parser.add_argument("--answers-dir", default="data/generation")
    evaluate_generation_parser.add_argument(
        "--legal-units-path",
        default="data/parsed/legal_units.json",
    )
    evaluate_generation_parser.add_argument(
        "--faithfulness-prompt-path",
        default="config/prompts/faithfulness_judge.txt",
    )
    evaluate_generation_parser.add_argument(
        "--correctness-prompt-path",
        default="config/prompts/correctness_judge.txt",
    )
    evaluate_generation_parser.add_argument(
        "--faithfulness-scores-path",
        default="data/evaluation/faithfulness_scores.json",
    )
    evaluate_generation_parser.add_argument(
        "--correctness-scores-path",
        default="data/evaluation/correctness_scores.json",
    )
    evaluate_generation_parser.add_argument(
        "--answer-evaluation-records-path",
        default="data/evaluation/generated_answer_evaluation.json",
    )
    evaluate_generation_parser.add_argument(
        "--generation-metrics-path",
        default="data/evaluation/generation_metrics.json",
    )
    evaluate_generation_parser.add_argument("--condition-ids")
    evaluate_generation_parser.add_argument("--query-ids")
    evaluate_generation_parser.add_argument("--overwrite", action="store_true")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run Stage 8 analysis and reporting.",
    )
    analyze_parser.add_argument("--config-path", default="config/config.yaml")
    analyze_parser.add_argument(
        "--retrieval-metrics-path",
        default="data/evaluation/retrieval_metrics.json",
    )
    analyze_parser.add_argument(
        "--retrieval-results-path",
        default="data/retrieval/retrieval_results.json",
    )
    analyze_parser.add_argument(
        "--chunk-characteristics-path",
        default="data/chunks/chunk_characteristics.json",
    )
    analyze_parser.add_argument("--chunks-dir", default="data/chunks")
    analyze_parser.add_argument(
        "--legal-units-path",
        default="data/parsed/legal_units.json",
    )
    analyze_parser.add_argument(
        "--context-coverage-metrics-path",
        default="data/evaluation/context_coverage_metrics.json",
    )
    analyze_parser.add_argument(
        "--context-assembly-records-path",
        default="data/evaluation/context_assembly_records.json",
    )
    analyze_parser.add_argument(
        "--generation-metrics-path",
        default="data/evaluation/generation_metrics.json",
    )
    analyze_parser.add_argument(
        "--generated-answer-evaluation-path",
        default="data/evaluation/generated_answer_evaluation.json",
    )
    analyze_parser.add_argument(
        "--xref-eligibility-report-path",
        default="data/dataset/xref_eligibility_report.json",
    )
    analyze_parser.add_argument("--output-dir", default="data/results")

    analyze_rq3_parser = subparsers.add_parser(
        "analyze-rq3",
        help="Analyze RQ3 cross-chunking efficiency without judge calls.",
    )
    analyze_rq3_parser.add_argument("--config-path", default="config/config.yaml")
    analyze_rq3_parser.add_argument(
        "--rq3-answers-dir", default="data/generation_rq3"
    )
    analyze_rq3_parser.add_argument(
        "--hierarchical-answers-path",
        default="data/generation/no_enrichment/answers.json",
    )
    analyze_rq3_parser.add_argument(
        "--rq3-context-records-path",
        default="data/evaluation/rq3_context_assembly_records.json",
    )
    analyze_rq3_parser.add_argument(
        "--hierarchical-context-records-path",
        default="data/evaluation/context_assembly_records.json",
    )
    analyze_rq3_parser.add_argument(
        "--retrieval-results-path",
        default="data/retrieval/retrieval_results.json",
    )
    analyze_rq3_parser.add_argument(
        "--output-dir", default="data/results/rq3_cross_chunking"
    )

    audit_parser = subparsers.add_parser(
        "audit",
        help="Create the Stage 7 manual judge-audit packet.",
    )
    audit_parser.add_argument("--config-path", default="config/config.yaml")
    audit_parser.add_argument(
        "--answer-evaluation-records-path",
        default="data/evaluation/generated_answer_evaluation.json",
    )
    audit_parser.add_argument("--answers-dir", default="data/generation")
    audit_parser.add_argument(
        "--faithfulness-scores-path",
        default="data/evaluation/faithfulness_scores.json",
    )
    audit_parser.add_argument(
        "--correctness-scores-path",
        default="data/evaluation/correctness_scores.json",
    )
    audit_parser.add_argument(
        "--output-csv-path",
        default="data/audit/manual_judge_audit_sample.csv",
    )
    audit_parser.add_argument(
        "--metadata-path",
        default="data/audit/manual_judge_audit_metadata.json",
    )
    audit_parser.add_argument("--sample-fraction", type=float)
    audit_parser.add_argument("--sample-size", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "fetch":
        fetch_discolqa(args.repo, args.branch, args.commit, Path(args.output_dir))
    elif args.stage == "parse":
        config = yaml.safe_load(Path(args.config_path).read_text(encoding="utf-8"))
        assert_snapshot_matches_config(Path(args.source_metadata_path), config)
        # Parse first so imports can validate gold IDs.
        legal_units, _, _ = write_corpus_outputs(
            raw_documents_dir=Path(args.raw_documents_dir),
            legal_units_path=Path(args.legal_units_path),
            canonical_texts_path=Path(args.canonical_texts_path),
            validation_report_path=Path(args.validation_report_path),
        )
        _, import_report = import_q4eu(
            evaluate_path=Path(args.evaluate_path),
            source_metadata_path=Path(args.source_metadata_path),
            output_path=Path(args.queries_path),
            report_path=Path(args.import_report_path),
            legal_units_path=Path(args.legal_units_path),
        )
        xref_graph = write_xref_outputs(
            legal_units_path=Path(args.legal_units_path),
            xref_graph_path=Path(args.xref_graph_path),
            validation_report_path=Path(args.validation_report_path),
        )
        _, eligibility_report = write_structural_eligibility_outputs(
            queries_path=Path(args.queries_path),
            xref_graph_path=Path(args.xref_graph_path),
            report_path=Path(args.xref_eligibility_report_path),
        )
        print(
            "Stage 1 parse complete: "
            f"{import_report['imported_question_count']} questions, "
            f"{len(legal_units)} legal units, "
            f"{xref_graph['summary']['edge_count']} xref edges, "
            f"{eligibility_report['structural_upper_bound_count']} structural xref candidates."
        )
    elif args.stage == "chunk":
        characteristics = write_chunk_outputs(
            config_path=Path(args.config_path),
            legal_units_path=Path(args.legal_units_path),
            canonical_texts_path=Path(args.canonical_texts_path),
            output_dir=Path(args.output_dir),
            characteristics_path=Path(args.characteristics_path),
            include_semantic=args.include_semantic,
        )
        print(
            "Stage 2 chunking complete: "
            f"{len(characteristics['configs'])} configurations."
        )
    elif args.stage == "index":
        characteristics = write_index_outputs(
            config_path=Path(args.config_path),
            chunks_dir=Path(args.chunks_dir),
            embeddings_dir=Path(args.embeddings_dir),
            indices_dir=Path(args.indices_dir),
            characteristics_path=Path(args.characteristics_path),
        )
        print(
            "Stage 3 indexing complete: "
            f"{len(characteristics['configs'])} configurations."
        )
    elif args.stage == "retrieve":
        records, _ = write_retrieval_outputs(
            config_path=Path(args.config_path),
            queries_path=Path(args.queries_path),
            chunks_dir=Path(args.chunks_dir),
            embeddings_dir=Path(args.embeddings_dir),
            indices_dir=Path(args.indices_dir),
            retrieval_results_path=Path(args.retrieval_results_path),
            retrieval_metrics_path=Path(args.retrieval_metrics_path),
        )
        print(f"Stage 4 retrieval evaluation complete: {len(records)} records.")
    elif args.stage == "assemble":
        records, _, xref_report = write_context_assembly_outputs(
            config_path=Path(args.config_path),
            queries_path=Path(args.queries_path),
            legal_units_path=Path(args.legal_units_path),
            chunks_dir=Path(args.chunks_dir),
            embeddings_dir=Path(args.embeddings_dir),
            indices_dir=Path(args.indices_dir),
            xref_graph_path=Path(args.xref_graph_path),
            retrieval_results_path=Path(args.retrieval_results_path),
            context_records_path=Path(args.context_records_path),
            coverage_metrics_path=Path(args.coverage_metrics_path),
            xref_eligibility_report_path=Path(args.xref_eligibility_report_path),
        )
        print(
            "Stage 5 context assembly complete: "
            f"{len(records)} records, "
            f"{xref_report['retrieval_conditioned_count']} xref-eligible queries."
        )
    elif args.stage == "assemble-rq3":
        records, _ = write_rq3_context_assembly_outputs(
            config_path=Path(args.config_path),
            queries_path=Path(args.queries_path),
            chunks_dir=Path(args.chunks_dir),
            retrieval_results_path=Path(args.retrieval_results_path),
            context_records_path=Path(args.context_records_path),
            context_metrics_path=Path(args.context_metrics_path),
        )
        print(f"RQ3 context assembly complete: {len(records)} records.")
    elif args.stage == "generate":
        summary = write_generation_outputs(
            config_path=Path(args.config_path),
            context_records_path=Path(args.context_records_path),
            coverage_metrics_path=Path(args.coverage_metrics_path),
            output_dir=Path(args.output_dir),
            prompt_path=Path(args.prompt_path),
            condition_ids=parse_csv(args.condition_ids),
            query_ids=parse_csv(args.query_ids),
            pilot_query_count=args.pilot_query_count,
            overwrite=args.overwrite,
        )
        print(
            "Stage 6 answer generation complete: "
            f"{summary['record_count']} selected records, "
            f"{sum(summary['generated_counts'].values())} generated, "
            f"{sum(summary['skipped_counts'].values())} skipped."
        )
    elif args.stage == "generate-rq3":
        summary = write_generation_outputs(
            config_path=Path(args.config_path),
            context_records_path=Path(args.context_records_path),
            coverage_metrics_path=Path(args.context_metrics_path),
            output_dir=Path(args.output_dir),
            prompt_path=Path(args.prompt_path),
            query_ids=parse_csv(args.query_ids),
            pilot_query_count=args.pilot_query_count,
            overwrite=args.overwrite,
            checkpoint_type="rq3_cross_chunking",
        )
        print(
            "RQ3 answer generation complete: "
            f"{summary['record_count']} selected records, "
            f"{sum(summary['generated_counts'].values())} generated, "
            f"{sum(summary['skipped_counts'].values())} skipped."
        )
    elif args.stage == "evaluate-generation":
        (
            faithfulness_scores,
            correctness_scores,
            _,
            _,
        ) = write_generated_answer_evaluation_outputs(
            config_path=Path(args.config_path),
            answers_dir=Path(args.answers_dir),
            legal_units_path=Path(args.legal_units_path),
            faithfulness_prompt_path=Path(args.faithfulness_prompt_path),
            correctness_prompt_path=Path(args.correctness_prompt_path),
            faithfulness_scores_path=Path(args.faithfulness_scores_path),
            correctness_scores_path=Path(args.correctness_scores_path),
            answer_evaluation_records_path=Path(args.answer_evaluation_records_path),
            generation_metrics_path=Path(args.generation_metrics_path),
            condition_ids=parse_evaluation_csv(args.condition_ids),
            query_ids=parse_evaluation_csv(args.query_ids),
            overwrite=args.overwrite,
        )
        print(
            "Stage 7 generated-answer evaluation complete: "
            f"{len(faithfulness_scores)} faithfulness records, "
            f"{len(correctness_scores)} correctness records."
        )
    elif args.stage == "analyze":
        summary = write_stage8_outputs(
            config_path=Path(args.config_path),
            retrieval_metrics_path=Path(args.retrieval_metrics_path),
            retrieval_results_path=Path(args.retrieval_results_path),
            chunk_characteristics_path=Path(args.chunk_characteristics_path),
            chunks_dir=Path(args.chunks_dir),
            context_coverage_metrics_path=Path(args.context_coverage_metrics_path),
            context_assembly_records_path=Path(args.context_assembly_records_path),
            legal_units_path=Path(args.legal_units_path),
            generation_metrics_path=Path(args.generation_metrics_path),
            generated_answer_evaluation_path=Path(args.generated_answer_evaluation_path),
            output_dir=Path(args.output_dir),
            xref_eligibility_report_path=Path(args.xref_eligibility_report_path),
        )
        print(
            "Stage 8 analysis complete: "
            f"{summary['retrieval']['target_acts_top_10_config_count']} retrieval configs, "
            f"{summary['generation']['zero_correctness_case_count']} zero-correctness cases."
        )
    elif args.stage == "analyze-rq3":
        summary = write_rq3_cross_chunking_outputs(
            config_path=Path(args.config_path),
            rq3_answers_dir=Path(args.rq3_answers_dir),
            hierarchical_answers_path=Path(args.hierarchical_answers_path),
            rq3_context_records_path=Path(args.rq3_context_records_path),
            hierarchical_context_records_path=Path(
                args.hierarchical_context_records_path
            ),
            retrieval_results_path=Path(args.retrieval_results_path),
            output_dir=Path(args.output_dir),
        )
        print(
            "RQ3 cross-chunking analysis complete: "
            f"{summary['query_count']} paired queries, "
            f"{summary['new_generation_count']} new answers."
        )
    elif args.stage == "audit":
        rows, _ = write_manual_audit_outputs(
            config_path=Path(args.config_path),
            answer_evaluation_records_path=Path(args.answer_evaluation_records_path),
            answers_dir=Path(args.answers_dir),
            faithfulness_scores_path=Path(args.faithfulness_scores_path),
            correctness_scores_path=Path(args.correctness_scores_path),
            output_csv_path=Path(args.output_csv_path),
            metadata_path=Path(args.metadata_path),
            sample_fraction=args.sample_fraction,
            sample_size=args.sample_size,
        )
        print(f"Manual audit sample complete: {len(rows)} records.")


if __name__ == "__main__":
    main()
