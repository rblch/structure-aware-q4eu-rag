from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import BadRequestError, OpenAI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking.tokenization import DEFAULT_ENCODING, count_tokens
from enrichment.rq3_context_assembly import (
    CHECKPOINT_TYPE as RQ3_CHECKPOINT_TYPE,
    assert_rq3_context_records,
    validate_rq3_config,
)
from utils.env import load_dotenv


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str


def openrouter_headers(gateway: dict[str, Any]) -> dict[str, str]:
    headers = {
        # Request serving-provider metadata.
        "X-OpenRouter-Metadata": "enabled",
    }
    if referer := os.environ.get(gateway.get("http_referer_env", "")):
        headers["HTTP-Referer"] = referer
    if title := os.environ.get(gateway.get("title_env", "")):
        headers["X-Title"] = title
    return headers


def response_routing_metadata(response: Any) -> dict[str, Any]:
    # Provider fields may be absent from gateway responses.
    return {
        "generation_id": getattr(response, "id", None),
        "provider": getattr(response, "provider", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
    }


class OpenRouterChatGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        load_dotenv()
        gateway = config["models"]["gateway"]
        generation_config = config["models"]["generation"]
        if generation_config["api"] != "chat_completions":
            raise ValueError(f"Unsupported generation API: {generation_config['api']}")

        api_key = os.environ.get(gateway["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{gateway['api_key_env']} is not set")

        headers = openrouter_headers(gateway)

        self.client = OpenAI(
            base_url=gateway["base_url"],
            api_key=api_key,
            default_headers=headers or None,
        )
        self.model = generation_config["model"]
        self.temperature = generation_config.get("temperature")
        self.max_tokens = generation_config["max_tokens"]

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        started = time.perf_counter()
        response, completion_kwargs = self._create_completion(messages)
        elapsed_seconds = time.perf_counter() - started
        choice = response.choices[0]
        return {
            "answer_text": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "response_model": response.model,
            "completion_kwargs": loggable_completion_kwargs(completion_kwargs),
            "usage": usage_to_dict(getattr(response, "usage", None)),
            "elapsed_seconds": elapsed_seconds,
            **response_routing_metadata(response),
        }

    def _create_completion(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[Any, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        for _ in range(3):
            try:
                return self.client.chat.completions.create(**kwargs), kwargs
            except BadRequestError as error:
                retry_kwargs = fallback_completion_kwargs(kwargs, str(error))
                if retry_kwargs == kwargs:
                    raise
                kwargs = retry_kwargs
        return self.client.chat.completions.create(**kwargs), kwargs


def write_generation_outputs(
    *,
    config_path: Path,
    context_records_path: Path,
    coverage_metrics_path: Path | None = None,
    output_dir: Path,
    prompt_path: Path,
    condition_ids: list[str] | None = None,
    query_ids: list[str] | None = None,
    pilot_query_count: int | None = None,
    overwrite: bool = False,
    checkpoint_type: str = "rq2_enrichment",
    encoding_name: str = DEFAULT_ENCODING,
) -> dict[str, Any]:
    if pilot_query_count and "pilot" not in output_dir.parts:
        raise ValueError(
            "--pilot-query-count requires a dedicated pilot output directory "
            "(e.g. data/generation/pilot); pilot answers must never share "
            f"files with the confirmatory run (got {output_dir})."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context_records = json.loads(context_records_path.read_text(encoding="utf-8"))
    if coverage_metrics_path is None:
        coverage_metrics_path = context_records_path.with_name(
            "context_coverage_metrics.json"
        )
    if checkpoint_type == "rq2_enrichment":
        assert_context_budget_checkpoint(
            config=config,
            context_records=context_records,
            coverage_metrics_path=coverage_metrics_path,
        )
    elif checkpoint_type == RQ3_CHECKPOINT_TYPE:
        assert_rq3_generation_checkpoint(
            config=config,
            context_records=context_records,
            context_records_path=context_records_path,
            context_metrics_path=coverage_metrics_path,
        )
    else:
        raise ValueError(f"Unsupported generation checkpoint: {checkpoint_type}")
    prompt = load_prompt_template(prompt_path)
    selected_records = select_context_records(
        context_records,
        condition_ids=condition_ids,
        query_ids=query_ids,
        pilot_query_count=pilot_query_count,
    )
    assert_context_window(
        selected_records,
        prompt=prompt,
        generation_config=config["models"]["generation"],
        encoding_name=encoding_name,
    )

    generator = OpenRouterChatGenerator(config)
    prompt_hash = file_sha256(prompt_path)
    existing_by_condition = load_existing_answers(output_dir)
    generated_counts: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}
    answers_by_condition: dict[str, list[dict[str, Any]]] = {
        condition_id: list(records)
        for condition_id, records in existing_by_condition.items()
    }

    for record in selected_records:
        condition_id = record["condition_id"]
        answers = answers_by_condition.setdefault(condition_id, [])
        existing = find_answer(answers, record["query_id"])
        if existing is not None:
            if not overwrite:
                assert_answer_record_current(
                    existing,
                    context_record=record,
                    prompt_sha256=prompt_hash,
                    generation_config=config["models"]["generation"],
                )
                skipped_counts[condition_id] = skipped_counts.get(condition_id, 0) + 1
                continue
            # Preserve answers outside an overwrite selection.
            answers.remove(existing)

        messages = render_messages(prompt, record)
        result = generator.generate(messages)
        assert_complete_generation(
            result,
            condition_id=condition_id,
            query_id=record["query_id"],
        )
        answers.append(
            build_answer_record(
                context_record=record,
                result=result,
                prompt_path=prompt_path,
                prompt_sha256=prompt_hash,
                generation_config=config["models"]["generation"],
                messages=messages,
                encoding_name=encoding_name,
            )
        )
        generated_counts[condition_id] = generated_counts.get(condition_id, 0) + 1
        write_condition_answers(output_dir, condition_id, answers)

    for condition_id, answers in sorted(answers_by_condition.items()):
        write_condition_answers(output_dir, condition_id, answers)

    return {
        "record_count": len(selected_records),
        "condition_count": len({record["condition_id"] for record in selected_records}),
        "output_dir": str(output_dir),
        "generated_counts": dict(sorted(generated_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
    }


def assert_context_budget_checkpoint(
    *,
    config: dict[str, Any],
    context_records: list[dict[str, Any]],
    coverage_metrics_path: Path,
) -> None:
    if not coverage_metrics_path.exists():
        raise RuntimeError(
            "Stage 6 requires the Stage 5 context-budget checkpoint at "
            f"{coverage_metrics_path}. Run Stage 5 successfully before generation."
        )
    metrics = json.loads(coverage_metrics_path.read_text(encoding="utf-8"))
    derivation = metrics.get("budget_derivation") or {}
    volume_match = metrics.get("volume_match") or {}
    configured_budget = config["enrichment"]["context_budget_tokens"]
    record_budgets = {
        record.get("context_budget_tokens") for record in context_records
    }
    checkpoint_matches = (
        derivation.get("matches_configured_budget") is True
        and derivation.get("configured_budget") == configured_budget
        and derivation.get("derived_budget") == configured_budget
        and metrics.get("context_budget_tokens") == configured_budget
        and record_budgets == {configured_budget}
        and volume_match.get("outside_tolerance_count") == 0
    )
    if not checkpoint_matches:
        raise RuntimeError(
            "Stage 6 blocked: the Stage 5 context-budget checkpoint, current "
            "config, context records, and volume-match result do not constitute "
            f"a valid checkpoint for the registered budget of {configured_budget} "
            "tokens. Rerun Stage 5 successfully and verify both "
            "budget_derivation.matches_configured_budget and the volume-match "
            "checkpoint before generation."
        )


def assert_rq3_generation_checkpoint(
    *,
    config: dict[str, Any],
    context_records: list[dict[str, Any]],
    context_records_path: Path,
    context_metrics_path: Path,
) -> None:
    if not context_metrics_path.exists():
        raise RuntimeError(
            f"RQ3 generation requires its assembly checkpoint at {context_metrics_path}"
        )
    metrics = json.loads(context_metrics_path.read_text(encoding="utf-8"))
    rq3_config = validate_rq3_config(config)
    expected_digest = hashlib.sha256(context_records_path.read_bytes()).hexdigest()
    checkpoint_matches = (
        metrics.get("checkpoint_type") == RQ3_CHECKPOINT_TYPE
        and metrics.get("ready_for_generation") is True
        and metrics.get("rq3_config") == rq3_config
        and metrics.get("record_count") == len(context_records)
        and metrics.get("context_records_sha256") == expected_digest
    )
    if not checkpoint_matches:
        raise RuntimeError(
            "RQ3 generation blocked: its context records, metrics, and current "
            "configuration do not form a valid assembly checkpoint. Rerun "
            "assemble-rq3 before generation."
        )
    assert_rq3_context_records(context_records, rq3_config)


def load_prompt_template(prompt_path: Path) -> PromptTemplate:
    text = prompt_path.read_text(encoding="utf-8").strip()
    user_marker = "\nUser:\n"
    if not text.startswith("System:\n") or user_marker not in text:
        raise ValueError(f"Prompt file has invalid System/User sections: {prompt_path}")
    system_text, user_text = text[len("System:\n") :].split(user_marker, 1)
    return PromptTemplate(system=system_text.strip(), user=user_text.strip())


def select_context_records(
    records: list[dict[str, Any]],
    *,
    condition_ids: list[str] | None,
    query_ids: list[str] | None,
    pilot_query_count: int | None,
) -> list[dict[str, Any]]:
    selected_query_ids = list(query_ids) if query_ids else None
    if selected_query_ids is None and pilot_query_count:
        selected_query_ids = sorted({record["query_id"] for record in records})[
            :pilot_query_count
        ]
    selected_conditions = set(condition_ids) if condition_ids else None
    selected_queries = set(selected_query_ids) if selected_query_ids else None

    selected = [
        record
        for record in records
        if (selected_conditions is None or record["condition_id"] in selected_conditions)
        and (selected_queries is None or record["query_id"] in selected_queries)
    ]
    return sorted(selected, key=lambda record: (record["query_id"], record["condition_id"]))


def assert_context_window(
    records: list[dict[str, Any]],
    *,
    prompt: PromptTemplate,
    generation_config: dict[str, Any],
    encoding_name: str,
) -> None:
    context_window = generation_config.get("context_window_tokens")
    if context_window is None:
        raise ValueError("models.generation.context_window_tokens must be configured")
    max_tokens = generation_config["max_tokens"]
    largest_required = 0
    for record in records:
        messages = render_messages(prompt, record)
        prompt_tokens = sum(
            count_tokens(message["content"], encoding_name) for message in messages
        )
        largest_required = max(largest_required, prompt_tokens + max_tokens)
    if largest_required > context_window:
        raise ValueError(
            "Generation prompt plus output budget exceeds context window: "
            f"{largest_required} > {context_window}"
        )


def load_existing_answers(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    answers: dict[str, list[dict[str, Any]]] = {}
    if not output_dir.exists():
        return answers
    for answers_path in output_dir.glob("*/answers.json"):
        answers[answers_path.parent.name] = json.loads(
            answers_path.read_text(encoding="utf-8")
        )
    return answers


def assert_complete_generation(
    result: dict[str, Any],
    *,
    condition_id: str,
    query_id: str,
) -> None:
    # Reject empty or truncated answers before judging.
    if result["finish_reason"] != "stop" or not result["answer_text"].strip():
        raise RuntimeError(
            f"Generation for condition {condition_id!r}, query {query_id!r} "
            f"returned finish_reason={result['finish_reason']!r} with "
            f"{len(result['answer_text'])} characters; increase "
            "models.generation.max_tokens (or investigate the provider "
            "response) and rerun."
        )


def find_answer(
    answers: list[dict[str, Any]],
    query_id: str,
) -> dict[str, Any] | None:
    return next(
        (answer for answer in answers if answer["query_id"] == query_id), None
    )


def assert_answer_record_current(
    existing: dict[str, Any],
    *,
    context_record: dict[str, Any],
    prompt_sha256: str,
    generation_config: dict[str, Any],
) -> None:
    # Reuse answers only when all generation inputs match.
    mismatched = [
        field
        for field, expected in (
            ("prompt_sha256", prompt_sha256),
            ("generation_model", generation_config["model"]),
            (
                "configured_completion_kwargs",
                configured_generation_kwargs(generation_config),
            ),
            ("question", context_record["question"]),
            ("context_text", context_record["context_text"]),
        )
        if existing.get(field) != expected
    ]
    if mismatched:
        raise RuntimeError(
            "Existing answer for condition "
            f"{existing['condition_id']!r}, query {existing['query_id']!r} was "
            f"generated with different inputs ({', '.join(mismatched)}). "
            "Rerun with --overwrite to regenerate."
        )


def render_messages(
    prompt: PromptTemplate,
    context_record: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt.system},
        {
            "role": "user",
            "content": prompt.user.format(
                context=context_record["context_text"],
                question=context_record["question"],
            ),
        },
    ]


def build_answer_record(
    *,
    context_record: dict[str, Any],
    result: dict[str, Any],
    prompt_path: Path,
    prompt_sha256: str,
    generation_config: dict[str, Any],
    messages: list[dict[str, str]],
    encoding_name: str,
) -> dict[str, Any]:
    prompt_token_count_estimate = sum(
        count_tokens(message["content"], encoding_name) for message in messages
    )
    completion_kwargs = result["completion_kwargs"]
    configured_completion_kwargs = configured_generation_kwargs(generation_config)
    return {
        "query_id": context_record["query_id"],
        "question": context_record["question"],
        "specificity": context_record["specificity"],
        "target_document_codes": context_record["target_document_codes"],
        "gold_unit_ids": context_record["gold_unit_ids"],
        "condition_id": context_record["condition_id"],
        "base_config_id": context_record["base_config_id"],
        "search_scope": context_record["search_scope"],
        "base_top_k": context_record["base_top_k"],
        "context_budget_tokens": context_record["context_budget_tokens"],
        "context_token_count": context_record["context_token_count"],
        "included_chunk_count": context_record["included_chunk_count"],
        "excluded_chunk_count": context_record["excluded_chunk_count"],
        "context_gold_recall": context_record["context_gold_recall"],
        "context_gold_precision": context_record["context_gold_precision"],
        "context_gold_f1": context_record["context_gold_f1"],
        "context_answer_unit_ids": context_record["context_answer_unit_ids"],
        "context_relevant_gold_unit_ids": context_record[
            "context_relevant_gold_unit_ids"
        ],
        "context_missing_gold_unit_ids": context_record["context_missing_gold_unit_ids"],
        "included_chunks": context_record["included_chunks"],
        "context_text": context_record["context_text"],
        "answer_text": result["answer_text"],
        "finish_reason": result["finish_reason"],
        "generation_model": generation_config["model"],
        "response_model": result["response_model"],
        "generation_id": result.get("generation_id"),
        "provider": result.get("provider"),
        "system_fingerprint": result.get("system_fingerprint"),
        "temperature": completion_kwargs.get("temperature"),
        "configured_temperature": generation_config.get("temperature"),
        "max_tokens": completion_kwargs.get("max_tokens"),
        "max_completion_tokens": completion_kwargs.get("max_completion_tokens"),
        "configured_max_tokens": generation_config["max_tokens"],
        "completion_kwargs": completion_kwargs,
        "configured_completion_kwargs": configured_completion_kwargs,
        "completion_kwargs_fallback_applied": (
            completion_kwargs != configured_completion_kwargs
        ),
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "prompt_token_count_estimate": prompt_token_count_estimate,
        "usage": result["usage"],
        "estimated_cost_usd": estimated_cost(result["usage"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_condition_answers(
    output_dir: Path,
    condition_id: str,
    answers: list[dict[str, Any]],
) -> None:
    path = output_dir / condition_id / "answers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(answers, key=lambda answer: answer["query_id"])
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


def fallback_completion_kwargs(
    kwargs: dict[str, Any],
    error_message: str,
) -> dict[str, Any]:
    lowered = error_message.lower()
    retry_kwargs = dict(kwargs)
    changed = False
    if "max_tokens" in lowered:
        retry_kwargs["max_completion_tokens"] = retry_kwargs.pop("max_tokens")
        changed = True
    if "temperature" in lowered:
        retry_kwargs.pop("temperature", None)
        changed = True
    return retry_kwargs if changed else kwargs


def configured_generation_kwargs(generation_config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": generation_config["model"],
        "max_tokens": generation_config["max_tokens"],
    }
    if generation_config.get("temperature") is not None:
        kwargs["temperature"] = generation_config["temperature"]
    return kwargs


def loggable_completion_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if key != "messages"
    }


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


def estimated_cost(usage: dict[str, Any] | None) -> float | None:
    if not usage:
        return None
    cost = usage.get("cost")
    return float(cost) if isinstance(cost, (int, float)) else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 6 answer generation.")
    parser.add_argument("--config-path", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--context-records-path",
        type=Path,
        default=Path("data/evaluation/context_assembly_records.json"),
    )
    parser.add_argument(
        "--coverage-metrics-path",
        type=Path,
        default=Path("data/evaluation/context_coverage_metrics.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/generation"))
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("config/prompts/answer_generation.txt"),
    )
    parser.add_argument("--condition-ids", help="Comma-separated condition IDs.")
    parser.add_argument("--query-ids", help="Comma-separated query IDs.")
    parser.add_argument("--pilot-query-count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_generation_outputs(
        config_path=args.config_path,
        context_records_path=args.context_records_path,
        coverage_metrics_path=args.coverage_metrics_path,
        output_dir=args.output_dir,
        prompt_path=args.prompt_path,
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


if __name__ == "__main__":
    main()
