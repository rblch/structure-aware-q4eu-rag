from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from openai import BadRequestError, OpenAI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_import.id_normalization import DOCUMENT_CODES, dedupe_preserving_order
from generation.answer_generation import (
    PromptTemplate,
    estimated_cost,
    fallback_completion_kwargs,
    file_sha256,
    load_prompt_template,
    openrouter_headers,
    response_routing_metadata,
    usage_to_dict,
)
from utils.env import load_dotenv
from utils.summary_stats import percentile


FAITHFULNESS_SCHEMA = {
    "faithfulness_score": "number from 0.0 to 1.0",
    "answer_abstains": "boolean",
    "abstention_justified": "boolean or null",
    "abstention_reason": "short string or empty string",
    "unfaithful_claims": "array of unsupported claim strings",
    "supporting_evidence": "array of short evidence strings from the context",
    "rationale": "short explanation string",
}

CORRECTNESS_SCHEMA = {
    "correctness_score": "number from 0.0 to 1.0",
    "answer_abstains": "boolean",
    "abstention_justified": "boolean or null",
    "abstention_reason": "short string or empty string",
    "material_omissions": "array of missing or weakly covered gold-text elements",
    "incorrect_claims": "array of incorrect or unsupported claim strings",
    "supporting_gold_evidence": "array of short evidence strings from the gold texts",
    "rationale": "short explanation string",
}

FAITHFULNESS_OUTPUT_SCHEMA_PATH = Path("config/schemas/faithfulness_output.json")
CORRECTNESS_OUTPUT_SCHEMA_PATH = Path("config/schemas/correctness_output.json")

JSON_TYPE_CHECKS = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "number": lambda value: isinstance(value, (int, float))
    and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}

CITATION_NUMBER_PATTERN = r"\d+[a-zA-Z]?(?:\.\d+[a-zA-Z]?)*(?:\s*\([0-9a-zA-Z]+\))*"
CITATION_RE = re.compile(
    rf"\b(?:(?P<document>{'|'.join(DOCUMENT_CODES)})\s+)?"
    r"(?P<kind>Art\.?|Article|Rec\.?|Recital)\s*"
    rf"(?P<number>{CITATION_NUMBER_PATTERN})",
    re.IGNORECASE,
)
# Parse explicit citation lists, but do not expand ranges.
CITATION_LIST_RE = re.compile(
    rf"\b(?:(?P<document>{'|'.join(DOCUMENT_CODES)})\s+)?"
    r"(?P<kind>Articles?|Arts?\.?|Recitals?|Recs?\.?)\s+"
    rf"(?P<body>{CITATION_NUMBER_PATTERN}"
    rf"(?:(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+){CITATION_NUMBER_PATTERN})+)",
    re.IGNORECASE,
)
CITATION_LIST_ITEM_RE = re.compile(CITATION_NUMBER_PATTERN)


class OpenRouterJudge:
    def __init__(
        self,
        config: dict[str, Any],
        evaluation_config: dict[str, Any],
    ) -> None:
        load_dotenv()
        gateway = config["models"]["gateway"]
        if evaluation_config["api"] != "chat_completions":
            raise ValueError(f"Unsupported evaluation API: {evaluation_config['api']}")

        api_key = os.environ.get(gateway["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{gateway['api_key_env']} is not set")

        headers = openrouter_headers(gateway)

        self.client = OpenAI(
            base_url=gateway["base_url"],
            api_key=api_key,
            default_headers=headers or None,
        )
        self.model = evaluation_config["model"]
        self.temperature = evaluation_config.get("temperature")
        self.max_tokens = evaluation_config["max_tokens"]

    def judge(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        usage_records: list[dict[str, Any] | None] = []
        retry_errors: list[str] = []
        retry_outputs: list[str] = []
        retry_messages = messages
        last_content = ""

        for attempt in range(3):
            response = self._create_completion(retry_messages)
            choice = response.choices[0]
            last_content = choice.message.content or ""
            usage_records.append(usage_to_dict(getattr(response, "usage", None)))
            try:
                if choice.finish_reason != "stop":
                    raise ValueError(
                        f"judge response finish_reason={choice.finish_reason!r}; "
                        "a complete response must finish with 'stop'"
                    )
                parsed = parse_json_object(last_content)
                validate_judge_output(parsed, output_schema)
            except (json.JSONDecodeError, ValueError) as error:
                retry_errors.append(str(error))
                retry_outputs.append(last_content)
                retry_messages = retry_json_messages(messages, last_content, error)
                continue

            return {
                "raw_judge_output": last_content,
                "parsed_judge_output": parsed,
                "finish_reason": choice.finish_reason,
                "response_model": response.model,
                "usage": sum_usage_records(usage_records),
                "elapsed_seconds": time.perf_counter() - started,
                "judge_attempt_count": attempt + 1,
                "judge_retry_errors": retry_errors,
                "judge_retry_outputs": retry_outputs,
                **response_routing_metadata(response),
            }

        raise RuntimeError(
            "Judge did not return schema-valid JSON after 3 attempts. "
            f"Last error: {retry_errors[-1] if retry_errors else 'unknown'}. "
            f"Last output preview: {last_content[:500]!r}"
        )

    def _create_completion(self, messages: list[dict[str, str]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        for _ in range(3):
            try:
                return self.client.chat.completions.create(**kwargs)
            except BadRequestError as error:
                retry_kwargs = fallback_completion_kwargs(kwargs, str(error))
                if retry_kwargs == kwargs:
                    raise
                kwargs = retry_kwargs
        return self.client.chat.completions.create(**kwargs)


def write_generated_answer_evaluation_outputs(
    *,
    config_path: Path,
    answers_dir: Path,
    legal_units_path: Path,
    faithfulness_prompt_path: Path,
    correctness_prompt_path: Path,
    faithfulness_scores_path: Path,
    correctness_scores_path: Path,
    answer_evaluation_records_path: Path,
    generation_metrics_path: Path,
    condition_ids: list[str] | None = None,
    query_ids: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["analysis"].get("enable_usefulness_judge"):
        raise NotImplementedError(
            "analysis.enable_usefulness_judge is set but no usefulness judge is "
            "implemented; set it to false."
        )
    answer_records = select_answer_records(
        load_answer_records(answers_dir),
        condition_ids=condition_ids,
        query_ids=query_ids,
    )
    legal_units_by_id = load_legal_units_by_id(legal_units_path)
    faithfulness_prompt = load_prompt_template(faithfulness_prompt_path)
    faithfulness_prompt_hash = file_sha256(faithfulness_prompt_path)
    correctness_prompt = load_prompt_template(correctness_prompt_path)
    correctness_prompt_hash = file_sha256(correctness_prompt_path)
    faithfulness_config = evaluation_config_for(config, "faithfulness")
    correctness_config = evaluation_config_for(config, "correctness")

    faithfulness_scores = (
        json.loads(faithfulness_scores_path.read_text(encoding="utf-8"))
        if faithfulness_scores_path.exists()
        else []
    )
    faithfulness_scores = normalize_existing_score_records(faithfulness_scores)
    correctness_scores = (
        json.loads(correctness_scores_path.read_text(encoding="utf-8"))
        if correctness_scores_path.exists()
        else []
    )
    correctness_scores = normalize_existing_score_records(correctness_scores)
    if overwrite:
        # Preserve scores outside an overwrite selection.
        selected_keys = {
            (answer["condition_id"], answer["query_id"]) for answer in answer_records
        }
        faithfulness_scores = without_selected_records(
            faithfulness_scores, selected_keys
        )
        correctness_scores = without_selected_records(correctness_scores, selected_keys)
    missing_faithfulness_answers = answers_missing_current_score(
        answer_records,
        existing_scores=keyed_records(faithfulness_scores),
        render_messages=lambda answer: render_faithfulness_messages(
            faithfulness_prompt, answer
        ),
        prompt_sha256=faithfulness_prompt_hash,
        judge_model=faithfulness_config["model"],
        score_kind="faithfulness",
    )
    missing_correctness_answers = answers_missing_current_score(
        answer_records,
        existing_scores=keyed_records(correctness_scores),
        render_messages=lambda answer: render_correctness_messages(
            correctness_prompt, answer, legal_units_by_id
        ),
        prompt_sha256=correctness_prompt_hash,
        judge_model=correctness_config["model"],
        score_kind="correctness",
    )
    faithfulness_judge = (
        OpenRouterJudge(config, faithfulness_config)
        if missing_faithfulness_answers
        else None
    )
    correctness_judge = (
        OpenRouterJudge(config, correctness_config)
        if missing_correctness_answers
        else None
    )
    faithfulness_schema = load_output_schema(FAITHFULNESS_OUTPUT_SCHEMA_PATH)
    correctness_schema = load_output_schema(CORRECTNESS_OUTPUT_SCHEMA_PATH)

    for answer in missing_faithfulness_answers:
        messages = render_faithfulness_messages(faithfulness_prompt, answer)
        if faithfulness_judge is None:
            raise RuntimeError("Faithfulness judge client was not initialized")
        result = faithfulness_judge.judge(messages, faithfulness_schema)
        faithfulness_scores.append(
            build_faithfulness_score_record(
                answer=answer,
                result=result,
                prompt_path=faithfulness_prompt_path,
                prompt_sha256=faithfulness_prompt_hash,
                input_sha256=judge_input_sha256(messages),
                evaluation_config=faithfulness_config,
            )
        )
        write_json(faithfulness_scores_path, sorted_score_records(faithfulness_scores))

    for answer in missing_correctness_answers:
        messages = render_correctness_messages(
            correctness_prompt,
            answer,
            legal_units_by_id,
        )
        if correctness_judge is None:
            raise RuntimeError("Correctness judge client was not initialized")
        result = correctness_judge.judge(messages, correctness_schema)
        correctness_scores.append(
            build_correctness_score_record(
                answer=answer,
                result=result,
                prompt_path=correctness_prompt_path,
                prompt_sha256=correctness_prompt_hash,
                input_sha256=judge_input_sha256(messages),
                evaluation_config=correctness_config,
            )
        )
        write_json(correctness_scores_path, sorted_score_records(correctness_scores))

    faithfulness_scores = sorted_score_records(faithfulness_scores)
    correctness_scores = sorted_score_records(correctness_scores)
    missing_score_keys: list[tuple[str, str]] = []
    answer_evaluation_records = build_answer_evaluation_records(
        answers=answer_records,
        faithfulness_scores=faithfulness_scores,
        correctness_scores=correctness_scores,
        missing_score_keys=missing_score_keys,
        legal_units_by_id=legal_units_by_id,
    )
    if missing_score_keys:
        # Full runs require identical query sets across conditions.
        if condition_ids is None and query_ids is None:
            raise RuntimeError(
                "Answers are missing judge scores for "
                f"{len(missing_score_keys)} (condition, query) pairs: "
                f"{missing_score_keys[:10]}"
            )
        print(
            f"WARNING: {len(missing_score_keys)} answers lack a faithfulness "
            "or correctness score and are excluded from evaluation records: "
            f"{missing_score_keys[:10]}"
        )
    metrics = summarize_generation_metrics(
        answers=answer_records,
        faithfulness_scores=faithfulness_scores,
        correctness_scores=correctness_scores,
        answer_evaluation_records=answer_evaluation_records,
        unfaithful_threshold=config["analysis"]["unfaithful_threshold"],
    )
    metrics["missing_score_pairs"] = [list(key) for key in missing_score_keys]
    write_json(faithfulness_scores_path, faithfulness_scores)
    write_json(correctness_scores_path, correctness_scores)
    if condition_ids is None and query_ids is None:
        write_json(answer_evaluation_records_path, answer_evaluation_records)
        write_json(generation_metrics_path, metrics)
    else:
        # Do not replace full-run aggregates with filtered statistics.
        print(
            "NOTE: filtered evaluation run — shared aggregate files were not "
            f"updated. Run an unfiltered evaluation to refresh "
            f"{answer_evaluation_records_path} and {generation_metrics_path}."
        )
    return faithfulness_scores, correctness_scores, answer_evaluation_records, metrics


def evaluation_config_for(
    config: dict[str, Any],
    score_kind: str,
) -> dict[str, Any]:
    evaluation_configs = config.get("models", {}).get("evaluation", {})
    evaluation_config = evaluation_configs.get(score_kind)
    if not isinstance(evaluation_config, dict):
        raise ValueError(
            f"models.evaluation.{score_kind} must be configured independently"
        )
    return evaluation_config


def load_answer_records(answers_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for answers_path in sorted(answers_dir.glob("*/answers.json")):
        records.extend(json.loads(answers_path.read_text(encoding="utf-8")))
    if not records:
        raise ValueError(f"No answers found under {answers_dir}")
    return records


def load_legal_units_by_id(legal_units_path: Path) -> dict[str, dict[str, Any]]:
    legal_units = json.loads(legal_units_path.read_text(encoding="utf-8"))
    return {unit["unit_id"]: unit for unit in legal_units}


def without_selected_records(
    records: list[dict[str, Any]],
    selected_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if (record["condition_id"], record["query_id"]) not in selected_keys
    ]


def judge_input_sha256(messages: list[dict[str, str]]) -> str:
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def answers_missing_current_score(
    answer_records: list[dict[str, Any]],
    *,
    existing_scores: dict[tuple[str, str], dict[str, Any]],
    render_messages: Callable[[dict[str, Any]], list[dict[str, str]]],
    prompt_sha256: str,
    judge_model: str,
    score_kind: str,
) -> list[dict[str, Any]]:
    # Reuse scores only when the model and rendered messages match.
    missing: list[dict[str, Any]] = []
    for answer in answer_records:
        existing = existing_scores.get((answer["condition_id"], answer["query_id"]))
        if existing is None:
            missing.append(answer)
            continue
        mismatched = [
            field
            for field, expected in (
                ("answer_text", answer["answer_text"]),
                ("prompt_sha256", prompt_sha256),
                ("judge_model", judge_model),
                ("judge_input_sha256", judge_input_sha256(render_messages(answer))),
            )
            if existing.get(field) != expected
        ]
        if mismatched:
            raise RuntimeError(
                f"Existing {score_kind} score for condition "
                f"{answer['condition_id']!r}, query {answer['query_id']!r} was "
                f"judged with different inputs ({', '.join(mismatched)}). "
                "Rerun with --overwrite to re-judge."
            )
    return missing


def normalize_existing_score_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        normalized_record = dict(record)
        normalized_record.setdefault("judge_attempt_count", 1)
        normalized_record.setdefault("judge_retry_errors", [])
        normalized_record.setdefault("judge_retry_outputs", [])
        normalized.append(normalized_record)
    return normalized


def retry_json_messages(
    messages: list[dict[str, str]],
    invalid_output: str,
    error: Exception,
) -> list[dict[str, str]]:
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "Your previous response could not be accepted because it did "
                f"not satisfy the required JSON schema and scoring rubric: {error}. "
                "Re-evaluate the same answer and return one corrected JSON object "
                "only, with no markdown or prose outside the JSON. Preserve the "
                "required field types and score only in the permitted increments.\n\n"
                f"Previous invalid response:\n{invalid_output[:2000]}"
            ),
        },
    ]


def sum_usage_records(
    usage_records: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    concrete_records = [record for record in usage_records if record]
    if not concrete_records:
        return None

    keys = {
        key
        for record in concrete_records
        for key, value in record.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    summed: dict[str, Any] = {
        key: sum(record.get(key, 0) or 0 for record in concrete_records)
        for key in sorted(keys)
    }
    last_record = concrete_records[-1]
    for key, value in last_record.items():
        if key not in summed:
            summed[key] = value
    return summed


def select_answer_records(
    records: list[dict[str, Any]],
    *,
    condition_ids: list[str] | None,
    query_ids: list[str] | None,
) -> list[dict[str, Any]]:
    selected_conditions = set(condition_ids) if condition_ids else None
    selected_queries = set(query_ids) if query_ids else None
    return sorted(
        [
            record
            for record in records
            if (
                selected_conditions is None
                or record["condition_id"] in selected_conditions
            )
            and (selected_queries is None or record["query_id"] in selected_queries)
        ],
        key=lambda record: (record["query_id"], record["condition_id"]),
    )


def render_faithfulness_messages(
    prompt: PromptTemplate,
    answer: dict[str, Any],
) -> list[dict[str, str]]:
    schema_instruction = (
        "Return only a JSON object with this schema:\n"
        f"{json.dumps(FAITHFULNESS_SCHEMA, indent=2)}"
    )
    return [
        {"role": "system", "content": prompt.system},
        {
            "role": "user",
            "content": (
                prompt.user.format(
                    question=answer["question"],
                    context=answer["context_text"],
                    answer=answer["answer_text"],
                )
                + "\n\n"
                + schema_instruction
            ),
        },
    ]


def render_correctness_messages(
    prompt: PromptTemplate,
    answer: dict[str, Any],
    legal_units_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    schema_instruction = (
        "Return only a JSON object with this schema:\n"
        f"{json.dumps(CORRECTNESS_SCHEMA, indent=2)}"
    )
    return [
        {"role": "system", "content": prompt.system},
        {
            "role": "user",
            "content": (
                prompt.user.format(
                    question=answer["question"],
                    gold_texts=render_gold_texts(answer, legal_units_by_id),
                    answer=answer["answer_text"],
                )
                + "\n\n"
                + schema_instruction
            ),
        },
    ]


def render_gold_texts(
    answer: dict[str, Any],
    legal_units_by_id: dict[str, dict[str, Any]],
) -> str:
    passages: list[str] = []
    for index, unit_id in enumerate(answer["gold_unit_ids"], start=1):
        unit = legal_units_by_id.get(unit_id)
        if unit is None:
            raise ValueError(
                f"Gold unit {unit_id!r} for {answer['query_id']} is not parsed"
            )
        passages.append(f"[G{index}] {unit_id}\n{unit['text']}")
    return "\n\n".join(passages)


def build_faithfulness_score_record(
    *,
    answer: dict[str, Any],
    result: dict[str, Any],
    prompt_path: Path,
    prompt_sha256: str,
    input_sha256: str,
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    parsed = result["parsed_judge_output"]
    faithfulness_score = bounded_score(parsed.get("faithfulness_score"))
    answer_abstains = bool(parsed.get("answer_abstains", False))
    abstention_justified = parsed.get("abstention_justified")
    abstention_justified = (
        bool(abstention_justified) if abstention_justified is not None else None
    )
    return {
        "query_id": answer["query_id"],
        "condition_id": answer["condition_id"],
        "question": answer["question"],
        "gold_unit_ids": answer["gold_unit_ids"],
        "context_answer_unit_ids": answer["context_answer_unit_ids"],
        "context_gold_recall": answer["context_gold_recall"],
        "answer_text": answer["answer_text"],
        "faithfulness_score": faithfulness_score,
        "answer_abstains": answer_abstains,
        "abstention_justified": abstention_justified,
        "abstention_reason": str(parsed.get("abstention_reason", "")),
        "unjustified_abstention": bool(
            answer_abstains and abstention_justified is False
        ),
        "unfaithful_claims": list(parsed.get("unfaithful_claims", [])),
        "supporting_evidence": list(parsed.get("supporting_evidence", [])),
        "rationale": str(parsed.get("rationale", "")),
        "judge_model": evaluation_config["model"],
        "response_model": result["response_model"],
        "judge_generation_id": result.get("generation_id"),
        "judge_provider": result.get("provider"),
        "judge_system_fingerprint": result.get("system_fingerprint"),
        "finish_reason": result["finish_reason"],
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "judge_input_sha256": input_sha256,
        "usage": result["usage"],
        "estimated_cost_usd": estimated_cost(result["usage"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "judge_attempt_count": result.get("judge_attempt_count", 1),
        "judge_retry_errors": result.get("judge_retry_errors", []),
        "judge_retry_outputs": result.get("judge_retry_outputs", []),
        "judged_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_judge_output": result["raw_judge_output"],
    }


def build_correctness_score_record(
    *,
    answer: dict[str, Any],
    result: dict[str, Any],
    prompt_path: Path,
    prompt_sha256: str,
    input_sha256: str,
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    parsed = result["parsed_judge_output"]
    correctness_score = bounded_score(parsed.get("correctness_score"))
    answer_abstains = bool(parsed.get("answer_abstains", False))
    abstention_justified = parsed.get("abstention_justified")
    abstention_justified = (
        bool(abstention_justified) if abstention_justified is not None else None
    )
    return {
        "query_id": answer["query_id"],
        "condition_id": answer["condition_id"],
        "question": answer["question"],
        "gold_unit_ids": answer["gold_unit_ids"],
        "context_answer_unit_ids": answer["context_answer_unit_ids"],
        "context_gold_recall": answer["context_gold_recall"],
        "answer_text": answer["answer_text"],
        "correctness_score": correctness_score,
        "answer_abstains": answer_abstains,
        "abstention_justified": abstention_justified,
        "abstention_reason": str(parsed.get("abstention_reason", "")),
        "unjustified_abstention": bool(
            answer_abstains and abstention_justified is False
        ),
        "material_omissions": list(parsed.get("material_omissions", [])),
        "incorrect_claims": list(parsed.get("incorrect_claims", [])),
        "supporting_gold_evidence": list(parsed.get("supporting_gold_evidence", [])),
        "rationale": str(parsed.get("rationale", "")),
        "judge_model": evaluation_config["model"],
        "response_model": result["response_model"],
        "judge_generation_id": result.get("generation_id"),
        "judge_provider": result.get("provider"),
        "judge_system_fingerprint": result.get("system_fingerprint"),
        "finish_reason": result["finish_reason"],
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "judge_input_sha256": input_sha256,
        "usage": result["usage"],
        "estimated_cost_usd": estimated_cost(result["usage"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "judge_attempt_count": result.get("judge_attempt_count", 1),
        "judge_retry_errors": result.get("judge_retry_errors", []),
        "judge_retry_outputs": result.get("judge_retry_outputs", []),
        "judged_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_judge_output": result["raw_judge_output"],
    }


def build_answer_evaluation_records(
    *,
    answers: list[dict[str, Any]],
    faithfulness_scores: list[dict[str, Any]],
    correctness_scores: list[dict[str, Any]],
    missing_score_keys: list[tuple[str, str]] | None = None,
    legal_units_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    faithfulness_by_key = keyed_records(faithfulness_scores)
    correctness_by_key = keyed_records(correctness_scores)
    records: list[dict[str, Any]] = []
    for answer in sorted_score_records(answers):
        key = (answer["condition_id"], answer["query_id"])
        faithfulness = faithfulness_by_key.get(key)
        correctness = correctness_by_key.get(key)
        if faithfulness is None or correctness is None:
            if missing_score_keys is not None:
                missing_score_keys.append(key)
            continue
        citation = citation_coverage(answer, legal_units_by_id)
        records.append(
            {
                "query_id": answer["query_id"],
                "condition_id": answer["condition_id"],
                "question": answer["question"],
                "gold_unit_ids": answer["gold_unit_ids"],
                "context_answer_unit_ids": answer["context_answer_unit_ids"],
                "context_gold_recall": answer["context_gold_recall"],
                "answer_text": answer["answer_text"],
                "generation_provider": answer.get("provider"),
                "faithfulness_judge_provider": faithfulness.get("judge_provider"),
                "correctness_judge_provider": correctness.get("judge_provider"),
                "faithfulness_score": faithfulness["faithfulness_score"],
                "correctness_score": correctness["correctness_score"],
                "answer_abstains": faithfulness["answer_abstains"],
                "faithfulness_abstention_justified": faithfulness[
                    "abstention_justified"
                ],
                "correctness_answer_abstains": correctness["answer_abstains"],
                "correctness_abstention_justified": correctness[
                    "abstention_justified"
                ],
                # Use either judge's flag for the anti-gaming gate.
                "either_judge_unjustified_abstention": bool(
                    faithfulness["unjustified_abstention"]
                    or correctness["unjustified_abstention"]
                ),
                **citation,
            }
        )
    return records


def keyed_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (record["condition_id"], record["query_id"]): record for record in records
    }


def citation_coverage(
    answer: dict[str, Any],
    legal_units_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    citations = extract_legal_citations(answer, legal_units_by_id)
    cited_unit_ids = dedupe_preserving_order(
        [
            citation["normalized_unit_id"]
            for citation in citations
            if citation["normalized_unit_id"] is not None
        ]
    )
    gold_unit_ids = set(answer["gold_unit_ids"])
    cited_unit_set = set(cited_unit_ids)
    cited_gold_unit_ids = sorted(gold_unit_ids & cited_unit_set)
    extra_citation_unit_ids = sorted(cited_unit_set - gold_unit_ids)
    missing_gold_citation_unit_ids = sorted(gold_unit_ids - cited_unit_set)
    precision = (
        len(cited_gold_unit_ids) / len(cited_unit_ids) if cited_unit_ids else 0.0
    )
    recall = len(cited_gold_unit_ids) / len(gold_unit_ids) if gold_unit_ids else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall > 0
        else 0.0
    )
    return {
        "citations": citations,
        "cited_unit_ids": cited_unit_ids,
        "cited_gold_unit_ids": cited_gold_unit_ids,
        "extra_citation_unit_ids": extra_citation_unit_ids,
        "missing_gold_citation_unit_ids": missing_gold_citation_unit_ids,
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "citation_count": len(citations),
        "unique_citation_count": len(cited_unit_ids),
        "extra_citation_count": len(extra_citation_unit_ids),
    }


def extract_legal_citations(
    answer: dict[str, Any],
    legal_units_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidate_codes = candidate_document_codes(answer)
    text = answer["answer_text"]
    positioned: list[tuple[int, dict[str, Any]]] = []
    list_spans: list[tuple[int, int]] = []
    for match in CITATION_LIST_RE.finditer(text):
        list_spans.append(match.span())
        kind = normalize_kind(match.group("kind"))
        for item in CITATION_LIST_ITEM_RE.finditer(match.group("body")):
            positioned.append(
                (
                    match.start("body") + item.start(),
                    build_citation(
                        raw=match.group(0),
                        kind=kind,
                        number=normalize_citation_number(item.group(0)),
                        document_code=match.group("document"),
                        candidate_codes=candidate_codes,
                        legal_units_by_id=legal_units_by_id,
                    ),
                )
            )
    for match in CITATION_RE.finditer(text):
        # Avoid re-extracting the first item of a matched list.
        if any(
            start <= match.start() and match.end() <= end
            for start, end in list_spans
        ):
            continue
        positioned.append(
            (
                match.start(),
                build_citation(
                    raw=match.group(0),
                    kind=normalize_kind(match.group("kind")),
                    number=normalize_citation_number(match.group("number")),
                    document_code=match.group("document"),
                    candidate_codes=candidate_codes,
                    legal_units_by_id=legal_units_by_id,
                ),
            )
        )
    positioned.sort(key=lambda entry: entry[0])
    return [citation for _, citation in positioned]


def build_citation(
    *,
    raw: str,
    kind: str,
    number: str,
    document_code: str | None,
    candidate_codes: set[str],
    legal_units_by_id: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    inferred = document_code is None
    if document_code is None:
        document_code = resolve_document_code(
            candidate_codes=candidate_codes,
            kind=kind,
            number=number,
            legal_units_by_id=legal_units_by_id,
        )
    normalized_unit_id = (
        f"{document_code} {kind}. {number}" if document_code is not None else None
    )
    return {
        "raw": raw.strip(),
        "normalized_unit_id": normalized_unit_id,
        "document_code_inferred": inferred and document_code is not None,
    }


def candidate_document_codes(answer: dict[str, Any]) -> set[str]:
    target_codes = set(answer.get("target_document_codes", []))
    gold_codes = {unit_id.split(" ", 1)[0] for unit_id in answer["gold_unit_ids"]}
    return target_codes or gold_codes


def resolve_document_code(
    *,
    candidate_codes: set[str],
    kind: str,
    number: str,
    legal_units_by_id: dict[str, dict[str, Any]] | None,
) -> str | None:
    if len(candidate_codes) == 1:
        return next(iter(candidate_codes))
    # Resolve an unqualified citation only when its act is unique.
    if legal_units_by_id is None:
        return None
    matching = [
        code
        for code in sorted(candidate_codes)
        if f"{code} {kind}. {number}" in legal_units_by_id
    ]
    return matching[0] if len(matching) == 1 else None


def normalize_kind(raw_kind: str) -> str:
    lowered = raw_kind.lower().rstrip(".")
    return "Art" if lowered.startswith("art") else "Rec"


def normalize_citation_number(raw_number: str) -> str:
    cleaned = re.sub(r"\s+", "", raw_number)
    return re.split(r"[.(]", cleaned, maxsplit=1)[0]


def summarize_generation_metrics(
    *,
    answers: list[dict[str, Any]],
    faithfulness_scores: list[dict[str, Any]],
    correctness_scores: list[dict[str, Any]],
    answer_evaluation_records: list[dict[str, Any]],
    unfaithful_threshold: float,
) -> dict[str, Any]:
    answers_by_key = {
        (answer["condition_id"], answer["query_id"]): answer for answer in answers
    }
    faithfulness_records = [
        score
        for score in faithfulness_scores
        if (score["condition_id"], score["query_id"]) in answers_by_key
    ]
    correctness_records = [
        score
        for score in correctness_scores
        if (score["condition_id"], score["query_id"]) in answers_by_key
    ]
    evaluation_records = [
        record
        for record in answer_evaluation_records
        if (record["condition_id"], record["query_id"]) in answers_by_key
    ]
    grouped_faithfulness = group_by_condition(faithfulness_records)
    grouped_correctness = group_by_condition(correctness_records)
    grouped_evaluations = group_by_condition(evaluation_records)
    grouped_answers = group_by_condition(answers)
    return {
        "record_count": len(evaluation_records),
        "faithfulness_record_count": len(faithfulness_records),
        "correctness_record_count": len(correctness_records),
        "unfaithful_threshold": unfaithful_threshold,
        "conditions": {
            condition_id: summarize_condition(
                answers=grouped_answers.get(condition_id, []),
                faithfulness_scores=grouped_faithfulness.get(condition_id, []),
                correctness_scores=grouped_correctness.get(condition_id, []),
                answer_evaluation_records=grouped_evaluations.get(condition_id, []),
                unfaithful_threshold=unfaithful_threshold,
            )
            for condition_id in sorted(grouped_answers)
        },
        "overall": summarize_condition(
            answers=answers,
            faithfulness_scores=faithfulness_records,
            correctness_scores=correctness_records,
            answer_evaluation_records=evaluation_records,
            unfaithful_threshold=unfaithful_threshold,
        ),
        # Retain per-query generation data for RQ3 latency joins.
        "generation_per_query": {
            condition_id: {
                answer["query_id"]: {
                    "elapsed_seconds": answer.get("elapsed_seconds"),
                    "prompt_tokens": (answer.get("usage") or {}).get("prompt_tokens"),
                    "completion_tokens": (answer.get("usage") or {}).get(
                        "completion_tokens"
                    ),
                    "total_tokens": (answer.get("usage") or {}).get("total_tokens"),
                    "estimated_cost_usd": answer.get("estimated_cost_usd"),
                }
                for answer in grouped_answers[condition_id]
            }
            for condition_id in sorted(grouped_answers)
        },
    }


def summarize_condition(
    *,
    answers: list[dict[str, Any]],
    faithfulness_scores: list[dict[str, Any]],
    correctness_scores: list[dict[str, Any]],
    answer_evaluation_records: list[dict[str, Any]],
    unfaithful_threshold: float,
) -> dict[str, Any]:
    return {
        "query_count": len(answer_evaluation_records),
        "faithfulness": {
            "mean": mean(
                [score["faithfulness_score"] for score in faithfulness_scores]
            ),
            "min": min_or_none(
                [score["faithfulness_score"] for score in faithfulness_scores]
            ),
            "max": max_or_none(
                [score["faithfulness_score"] for score in faithfulness_scores]
            ),
            "unfaithful_response_rate": mean(
                [
                    score["faithfulness_score"] < unfaithful_threshold
                    for score in faithfulness_scores
                ]
            ),
            "answer_abstention_rate": mean(
                [score["answer_abstains"] for score in faithfulness_scores]
            ),
            "unjustified_abstention_rate": mean(
                [score["unjustified_abstention"] for score in faithfulness_scores]
            ),
        },
        "correctness": {
            "mean": mean([score["correctness_score"] for score in correctness_scores]),
            "min": min_or_none(
                [score["correctness_score"] for score in correctness_scores]
            ),
            "max": max_or_none(
                [score["correctness_score"] for score in correctness_scores]
            ),
            "answer_abstention_rate": mean(
                [score["answer_abstains"] for score in correctness_scores]
            ),
            "unjustified_abstention_rate": mean(
                [score["unjustified_abstention"] for score in correctness_scores]
            ),
        },
        "citation_coverage": summarize_citation_coverage(answer_evaluation_records),
        "generation_efficiency": summarize_efficiency(answers),
        "faithfulness_judge_efficiency": summarize_efficiency(faithfulness_scores),
        "correctness_judge_efficiency": summarize_efficiency(correctness_scores),
        "judge_efficiency": summarize_efficiency(
            faithfulness_scores + correctness_scores
        ),
    }


def summarize_citation_coverage(
    answer_evaluation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "mean_precision": mean(
            [record["citation_precision"] for record in answer_evaluation_records]
        ),
        "mean_recall": mean(
            [
                record["citation_recall"]
                for record in answer_evaluation_records
                if record["citation_recall"] is not None
            ]
        ),
        "mean_f1": mean(
            [record["citation_f1"] for record in answer_evaluation_records]
        ),
        "mean_extra_citation_count": mean(
            [record["extra_citation_count"] for record in answer_evaluation_records]
        ),
        "mean_unique_citation_count": mean(
            [record["unique_citation_count"] for record in answer_evaluation_records]
        ),
    }


def summarize_efficiency(records: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [record.get("estimated_cost_usd") or 0.0 for record in records]
    elapsed = [record.get("elapsed_seconds") or 0.0 for record in records]
    prompt_tokens = [
        (record.get("usage") or {}).get("prompt_tokens", 0) for record in records
    ]
    completion_tokens = [
        (record.get("usage") or {}).get("completion_tokens", 0) for record in records
    ]
    total_tokens = [
        (record.get("usage") or {}).get("total_tokens", 0) for record in records
    ]
    return {
        "total_cost_usd": sum(costs),
        "mean_cost_usd": mean(costs),
        "total_elapsed_seconds": sum(elapsed),
        "mean_elapsed_seconds": mean(elapsed),
        "median_elapsed_seconds": percentile(elapsed, 50) if elapsed else None,
        "p95_elapsed_seconds": percentile(elapsed, 95) if elapsed else None,
        "total_prompt_tokens": sum(prompt_tokens),
        "mean_prompt_tokens": mean(prompt_tokens),
        "median_prompt_tokens": percentile(prompt_tokens, 50)
        if prompt_tokens
        else None,
        "p95_prompt_tokens": percentile(prompt_tokens, 95) if prompt_tokens else None,
        "total_completion_tokens": sum(completion_tokens),
        "mean_completion_tokens": mean(completion_tokens),
        "total_tokens": sum(total_tokens),
        "mean_total_tokens": mean(total_tokens),
        "median_total_tokens": percentile(total_tokens, 50) if total_tokens else None,
        "p95_total_tokens": percentile(total_tokens, 95) if total_tokens else None,
    }


def group_by_condition(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["condition_id"], []).append(record)
    return groups


def reject_json_constant(name: str) -> Any:
    # Reject non-finite JSON numbers before schema validation.
    raise ValueError(f"Non-standard JSON constant {name!r} is not allowed")


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped, parse_constant=reject_json_constant)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0), parse_constant=reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError("Judge output JSON is not an object")
    return value


def load_output_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def matches_json_type(value: Any, type_spec: str | list[str]) -> bool:
    types = type_spec if isinstance(type_spec, list) else [type_spec]
    return any(JSON_TYPE_CHECKS[json_type](value) for json_type in types)


def schema_violations(parsed: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    violations = [
        f"missing required field {field!r}"
        for field in schema["required"]
        if field not in parsed
    ]
    properties = schema["properties"]
    if not schema.get("additionalProperties", True):
        violations.extend(
            f"unexpected field {field!r}" for field in parsed if field not in properties
        )
    for field, spec in properties.items():
        if field not in parsed:
            continue
        value = parsed[field]
        if not matches_json_type(value, spec["type"]):
            violations.append(
                f"field {field!r} must be {spec['type']}, got {type(value).__name__}"
            )
            continue
        if isinstance(value, float) and not math.isfinite(value):
            violations.append(f"field {field!r} must be a finite number")
            continue
        if "minimum" in spec and value < spec["minimum"]:
            violations.append(f"field {field!r} is below minimum {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            violations.append(f"field {field!r} is above maximum {spec['maximum']}")
        if "multipleOf" in spec and isinstance(value, (int, float)):
            increment = float(spec["multipleOf"])
            quotient = float(value) / increment
            if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                violations.append(
                    f"field {field!r} must be a multiple of {increment}, got {value}"
                )
        if isinstance(value, list) and "items" in spec:
            violations.extend(
                f"field {field!r} item {index} must be {spec['items']['type']}, "
                f"got {type(item).__name__}"
                for index, item in enumerate(value)
                if not matches_json_type(item, spec["items"]["type"])
            )
    return violations


def validate_judge_output(parsed: dict[str, Any], schema: dict[str, Any]) -> None:
    violations = schema_violations(parsed, schema)
    # Abstentions require an explicit justification verdict.
    if parsed.get("answer_abstains") is True and not isinstance(
        parsed.get("abstention_justified"), bool
    ):
        violations.append(
            "abstention_justified must be true or false (not null) when "
            "answer_abstains is true"
        )
    if violations:
        raise ValueError(
            "Judge output failed schema validation: " + "; ".join(violations)
        )


def bounded_score(value: Any) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"Judge score is not a finite number: {value!r}")
    return max(0.0, min(1.0, score))


def mean(values: list[float | int | bool]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def sorted_score_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (record["query_id"], record["condition_id"]),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 7 generated-answer evaluation."
    )
    parser.add_argument("--config-path", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--answers-dir", type=Path, default=Path("data/generation"))
    parser.add_argument(
        "--legal-units-path",
        type=Path,
        default=Path("data/parsed/legal_units.json"),
    )
    parser.add_argument(
        "--faithfulness-prompt-path",
        type=Path,
        default=Path("config/prompts/faithfulness_judge.txt"),
    )
    parser.add_argument(
        "--correctness-prompt-path",
        type=Path,
        default=Path("config/prompts/correctness_judge.txt"),
    )
    parser.add_argument(
        "--faithfulness-scores-path",
        type=Path,
        default=Path("data/evaluation/faithfulness_scores.json"),
    )
    parser.add_argument(
        "--correctness-scores-path",
        type=Path,
        default=Path("data/evaluation/correctness_scores.json"),
    )
    parser.add_argument(
        "--answer-evaluation-records-path",
        type=Path,
        default=Path("data/evaluation/generated_answer_evaluation.json"),
    )
    parser.add_argument(
        "--generation-metrics-path",
        type=Path,
        default=Path("data/evaluation/generation_metrics.json"),
    )
    parser.add_argument("--condition-ids")
    parser.add_argument("--query-ids")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    (
        faithfulness_scores,
        correctness_scores,
        _,
        _,
    ) = write_generated_answer_evaluation_outputs(
        config_path=args.config_path,
        answers_dir=args.answers_dir,
        legal_units_path=args.legal_units_path,
        faithfulness_prompt_path=args.faithfulness_prompt_path,
        correctness_prompt_path=args.correctness_prompt_path,
        faithfulness_scores_path=args.faithfulness_scores_path,
        correctness_scores_path=args.correctness_scores_path,
        answer_evaluation_records_path=args.answer_evaluation_records_path,
        generation_metrics_path=args.generation_metrics_path,
        condition_ids=parse_csv(args.condition_ids),
        query_ids=parse_csv(args.query_ids),
        overwrite=args.overwrite,
    )
    print(
        "Stage 7 generated-answer evaluation complete: "
        f"{len(faithfulness_scores)} faithfulness records, "
        f"{len(correctness_scores)} correctness records."
    )


if __name__ == "__main__":
    main()
