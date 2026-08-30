from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from evaluation.manual_audit import MANUAL_COLUMNS
from evaluation.manual_audit_review import (
    gold_reference_text,
    load_audit_rows,
    load_legal_units,
    load_reviews,
    merge_reviews,
    phase_one_complete,
    review_complete,
    review_order,
    reviewer_output_path,
    rows_to_csv,
    save_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review a manual audit packet.")
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=Path("data/audit/manual_judge_audit_sample.csv"),
    )
    parser.add_argument(
        "--legal-units-path",
        type=Path,
        default=Path("data/parsed/legal_units.json"),
    )
    parser.add_argument("--reviews-dir", type=Path, default=Path("data/audit/reviews"))
    return parser.parse_args()


def select_index(rows: list[dict[str, str]], reviews: dict[str, dict[str, str]]) -> int:
    for index, row in enumerate(rows):
        if not review_complete(reviews.get(row["audit_id"], {})):
            return index
    return 0


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Manual judge audit", layout="wide")
    st.title("Manual judge audit")
    st.caption("Local, blinded review of the generated audit sample")

    try:
        audit_rows = load_audit_rows(args.audit_path)
        legal_units = load_legal_units(args.legal_units_path)
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(str(error))
        st.stop()

    reviewer_id = st.sidebar.text_input(
        "Reviewer ID", help="Letters, numbers, dots, underscores, and hyphens only."
    ).strip()
    if not reviewer_id:
        st.info("Enter a reviewer ID in the sidebar to begin.")
        st.stop()
    try:
        output_path = reviewer_output_path(args.reviews_dir, reviewer_id)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    reviews = load_reviews(output_path)
    rows = review_order(audit_rows)
    completed_count = sum(
        review_complete(reviews.get(row["audit_id"], {})) for row in rows
    )
    st.sidebar.progress(completed_count / len(rows))
    st.sidebar.write(f"{completed_count} of {len(rows)} completed")

    state_key = f"audit_index_{reviewer_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = select_index(rows, reviews)
    index = min(st.session_state[state_key], len(rows) - 1)
    row = rows[index]
    existing = reviews.get(row["audit_id"], {})

    previous, position, next_item = st.columns([1, 3, 1])
    if previous.button("Previous", disabled=index == 0, use_container_width=True):
        st.session_state[state_key] = index - 1
        st.rerun()
    position.markdown(
        f"<p style='text-align:center'>Item {index + 1} of {len(rows)}</p>",
        unsafe_allow_html=True,
    )
    if next_item.button(
        "Next", disabled=index == len(rows) - 1, use_container_width=True
    ):
        st.session_state[state_key] = index + 1
        st.rerun()

    st.subheader("Question")
    st.write(row["question"])
    answer_column, reference_column = st.columns(2)
    with answer_column:
        st.subheader("Generated answer")
        st.text_area(
            "Generated answer",
            row["answer_text"],
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )
        st.subheader("Retrieved context")
        st.text_area(
            "Retrieved context",
            row["context_text"],
            height=420,
            disabled=True,
            label_visibility="collapsed",
        )
    with reference_column:
        st.subheader("Gold legal reference")
        st.text_area(
            "Gold legal reference",
            gold_reference_text(row, legal_units),
            height=745,
            disabled=True,
            label_visibility="collapsed",
        )

    if not phase_one_complete(existing):
        render_blinded_form(row, existing, reviewer_id, output_path)
    else:
        render_comparison_form(
            row, existing, reviewer_id, output_path, state_key, index, len(rows)
        )

    if completed_count == len(rows):
        st.success("Review complete.")
        merged = merge_reviews(audit_rows, load_reviews(output_path))
        st.download_button(
            "Download completed audit CSV",
            rows_to_csv(merged),
            file_name="manual_judge_audit_completed.csv",
            mime="text/csv",
        )


def render_blinded_form(
    row: dict[str, str],
    existing: dict[str, str],
    reviewer_id: str,
    output_path: Path,
) -> None:
    st.divider()
    st.subheader("Step 1 — Independent assessment")
    st.caption(
        "Automatic scores and rationales remain hidden until these ratings are saved. "
        "Scores use 0.0 (fully unsupported/incorrect) to 1.0 (fully supported/correct)."
    )
    with st.form(f"blinded_{row['audit_id']}"):
        faithfulness = st.select_slider(
            "Faithfulness to the retrieved context",
            options=[step / 10 for step in range(11)],
            value=float(existing.get("manual_faithfulness_score") or 1.0),
        )
        correctness = st.select_slider(
            "Correctness against the gold legal reference",
            options=[step / 10 for step in range(11)],
            value=float(existing.get("manual_correctness_score") or 1.0),
        )
        abstention = st.selectbox(
            "Was abstention handled correctly?",
            ["Yes", "No", "Not applicable", "Unclear"],
            index=choice_index(
                ["Yes", "No", "Not applicable", "Unclear"],
                existing.get("manual_abstention_handled_correctly", "Not applicable"),
            ),
        )
        submitted = st.form_submit_button("Save independent assessment")
    if submitted:
        review = review_template(row["audit_id"], existing, reviewer_id)
        review.update(
            {
                "manual_faithfulness_score": faithfulness,
                "manual_correctness_score": correctness,
                "manual_abstention_handled_correctly": abstention,
            }
        )
        save_review(output_path, review)
        st.rerun()


def render_comparison_form(
    row: dict[str, str],
    existing: dict[str, str],
    reviewer_id: str,
    output_path: Path,
    state_key: str,
    index: int,
    row_count: int,
) -> None:
    st.divider()
    st.subheader("Step 2 — Compare with the automatic judge")
    st.write(
        f"Your faithfulness score: **{existing['manual_faithfulness_score']}** · "
        f"Automatic: **{row['faithfulness_score']}**  \n"
        f"Your correctness score: **{existing['manual_correctness_score']}** · "
        f"Automatic: **{row['correctness_score']}**"
    )
    with st.expander("Automatic judge rationales", expanded=True):
        st.markdown("**Unsupported claims identified by the judge**")
        st.write(row["faithfulness_unfaithful_claims"] or "None recorded.")
        st.markdown("**Faithfulness rationale**")
        st.write(row["faithfulness_judge_rationale"] or "No rationale recorded.")
        st.markdown("**Correctness rationale**")
        st.write(row["correctness_judge_rationale"] or "No rationale recorded.")

    with st.form(f"comparison_{row['audit_id']}"):
        reasonable_choices = ["Yes", "No", "Unclear"]
        reasonable = st.selectbox(
            "Are the automatic judge scores reasonable?",
            reasonable_choices,
            index=choice_index(
                reasonable_choices,
                existing.get("manual_judge_score_reasonable", "Yes"),
            ),
        )
        missed_choices = ["No", "Yes", "Unclear"]
        missed = st.selectbox(
            "Did the automatic judge miss unsupported claims?",
            missed_choices,
            index=choice_index(
                missed_choices,
                existing.get("manual_unsupported_claims_missed", "No"),
            ),
        )
        notes = st.text_area("Notes (optional)", existing.get("manual_notes", ""))
        submitted = st.form_submit_button("Complete item")
    if submitted:
        review = review_template(row["audit_id"], existing, reviewer_id)
        review.update(
            {
                "manual_judge_score_reasonable": reasonable,
                "manual_unsupported_claims_missed": missed,
                "manual_notes": notes,
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        save_review(output_path, review)
        if index < row_count - 1:
            st.session_state[state_key] = index + 1
        st.rerun()


def review_template(
    audit_id: str, existing: dict[str, str], reviewer_id: str
) -> dict[str, str]:
    review = {column: existing.get(column, "") for column in MANUAL_COLUMNS}
    review.update({"audit_id": audit_id, "reviewer_id": reviewer_id})
    return review


def choice_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


if __name__ == "__main__":
    main()
