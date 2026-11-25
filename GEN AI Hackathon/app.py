"""Streamlit UI for Safe Public Comment Rewrite Reviewer (PS 215)."""

import io
import json
import random
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

import config
import data
import judge
import report
import similarity
import toxicity

st.set_page_config(
    page_title="Safe Public Comment Rewrite Reviewer",
    page_icon="🛡️",
    layout="wide",
)

BADGE_STYLES = {
    config.VERDICT_SAFE: ("#15803d", "#dcfce7", "#86efac"),
    config.VERDICT_MEANING_LOST: ("#c2410c", "#ffedd5", "#fdba74"),
    config.VERDICT_STILL_TOXIC: ("#b91c1c", "#fee2e2", "#fca5a5"),
}


@st.cache_resource
def load_models() -> tuple[Any, Any]:
    """Load Toxic-BERT and MiniLM models once per Streamlit session."""
    tox_model = toxicity.get_model()
    sim_model = similarity.get_model()
    return tox_model, sim_model


@st.cache_data
def load_masked_examples() -> list[dict[str, str]]:
    """Load a fixed sample of ParaDetox pairs with masked labels for the dropdown."""
    pairs = data.load_pairs()
    rng = random.Random(config.EVAL_RANDOM_SEED)
    sample_count = min(config.UI_EXAMPLE_COUNT, len(pairs))
    sampled = rng.sample(pairs, sample_count)

    examples: list[dict[str, str]] = []
    for idx, (orig, neut) in enumerate(sampled, start=1):
        masked_orig = toxicity.mask(orig)
        masked_neut = toxicity.mask(neut)
        short_orig = masked_orig if len(masked_orig) <= 70 else masked_orig[:67] + "..."
        short_neut = masked_neut if len(masked_neut) <= 55 else masked_neut[:52] + "..."
        examples.append(
            {
                "label": f"Example {idx}: {short_orig} -> {short_neut}",
                "original": orig,
                "candidate": neut,
            }
        )
    return examples


def _render_verdict_badge(label: str, explanation: str) -> None:
    """Display a large colored verdict badge and explanation."""
    text_color, bg_color, border_color = BADGE_STYLES.get(
        label, ("#334155", "#f1f5f9", "#cbd5e1")
    )
    st.markdown(
        f"""
        <div style="
            padding: 1.1rem 1.4rem;
            border-radius: 0.75rem;
            background-color: {bg_color};
            border: 2px solid {border_color};
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.85rem; font-weight: 600; color: {text_color}; text-transform: uppercase; letter-spacing: 0.05em;">
                Evaluation Verdict
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {text_color}; margin: 0.2rem 0;">
                {label}
            </div>
            <div style="font-size: 0.98rem; color: #1e293b;">
                {explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_charts_figure(result: dict[str, Any]) -> plt.Figure:
    """Build side-by-side percentage comparison charts for toxicity and meaning retention."""
    before_pct = float(result["toxicity_before"]["toxic"]) * 100.0
    after_pct = float(result["toxicity_after"]["toxic"]) * 100.0
    drop_pct = max(0.0, float(result["toxicity_drop"]) * 100.0)
    retention_pct = float(result["similarity"]) * 100.0
    threshold_pct = config.TOXIC_THRESHOLD * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=150)

    # Chart 1: Toxicity Before vs After (%)
    tox_labels = ["Original", "Candidate"]
    tox_values = [before_pct, after_pct]
    tox_colors = ["#ef4444" if before_pct > threshold_pct else "#64748b",
                  "#ef4444" if after_pct > threshold_pct else "#22c55e"]
    bars1 = ax1.bar(tox_labels, tox_values, color=tox_colors, width=0.5)
    ax1.axhline(
        threshold_pct,
        color="#f97316",
        linestyle="--",
        linewidth=1.4,
        label=f"Safe limit ({threshold_pct:.0f}%)",
    )
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Toxicity Comparison (%)", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, val in zip(bars1, tox_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Chart 2: Meaning Retention & Toxicity Reduction (%)
    ret_labels = ["Meaning Retention\n(MiniLM)", "Toxicity Drop"]
    ret_values = [retention_pct, drop_pct]
    ret_colors = ["#3b82f6", "#10b981"]
    bars2 = ax2.bar(ret_labels, ret_values, color=ret_colors, width=0.5)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Meaning Retention & Reduction (%)", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, val in zip(bars2, ret_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def _pdf_safe(text: str) -> str:
    """Convert text to latin-1 safe string for FPDF built-in Helvetica font."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _build_pdf_bytes(result: dict[str, Any], fig: plt.Figure) -> bytes:
    """Generate a masked PDF evaluation report with embedded percentage charts."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    page_w = pdf.epw

    def _write_line(text: str, size: int = 10, style: str = "", height: float = 5.5) -> None:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(page_w, height, _pdf_safe(text), new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(0, 0, 0)
    _write_line("Safe Public Comment Rewrite Reviewer - Report", size=16, style="B", height=9)
    pdf.set_text_color(90, 90, 90)
    _write_line("PS 215 Evaluation Summary (All offensive terms masked)", size=10, height=6)
    pdf.ln(3)

    # Verdict
    verdict_info = result["verdict"]
    pdf.set_text_color(0, 0, 0)
    _write_line(f"Verdict: {verdict_info['label']}", size=13, style="B", height=8)
    _write_line(verdict_info["explanation"], size=10, height=5.5)
    pdf.ln(3)

    # Percentage Metrics
    tox_b = result["toxicity_before"]
    tox_a = result["toxicity_after"]
    before_pct = float(tox_b["toxic"]) * 100.0
    after_pct = float(tox_a["toxic"]) * 100.0
    drop_pct = float(result["toxicity_drop"]) * 100.0
    sim_pct = float(result["similarity"]) * 100.0

    _write_line("Key Metrics (%)", size=11, style="B", height=7)
    _write_line(
        f"- Original Toxicity: {before_pct:.1f}% "
        f"(top: {tox_b['top_category']} {float(tox_b['top_score']) * 100:.1f}%)"
    )
    _write_line(
        f"- Candidate Toxicity: {after_pct:.1f}% "
        f"(top: {tox_a['top_category']} {float(tox_a['top_score']) * 100:.1f}%)"
    )
    _write_line(f"- Toxicity Drop: {drop_pct:.1f}%")
    _write_line(f"- Meaning Retention (MiniLM Similarity): {sim_pct:.1f}%")
    pdf.ln(3)

    # Embed chart image
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
    img_buf.seek(0)
    pdf.set_x(pdf.l_margin)
    pdf.image(img_buf, x=pdf.l_margin, w=page_w)
    pdf.ln(4)

    # Gemini Judge
    judge_info = result["judge"]
    _write_line("Gemini Meaning Preservation Judge", size=11, style="B", height=7)
    status_str = "Available" if judge_info["available"] else "Unavailable (fallback used)"
    _write_line(f"- Status: {status_str}")
    _write_line(f"- Meaning Preserved: {judge_info['meaning_preserved']}")
    _write_line(f"- What Was Lost: {judge_info['what_was_lost']}")
    _write_line(f"- Reason: {judge_info['reason']}")
    pdf.ln(3)

    # Masked texts
    _write_line("Masked Comment Comparison", size=11, style="B", height=7)
    _write_line(f"Original (masked): {result['masked_original']}")
    pdf.ln(1)
    _write_line(f"Candidate (masked): {result['masked_candidate']}")

    return bytes(pdf.output())


def main() -> None:
    """Render the Safe Public Comment Rewrite Reviewer application."""
    st.title("Safe Public Comment Rewrite Reviewer")
    st.caption(
        "Evaluation tool (PS 215): checks whether a candidate rewrite reduces toxicity "
        "while preserving core meaning, without repeating harmful language in outputs."
    )

    with st.spinner("Loading evaluation models..."):
        load_models()

    examples = load_masked_examples()
    option_labels = ["Custom input..."] + [ex["label"] for ex in examples]
    label_to_example = {ex["label"]: ex for ex in examples}

    if "original_input" not in st.session_state:
        st.session_state["original_input"] = ""
    if "candidate_input" not in st.session_state:
        st.session_state["candidate_input"] = ""

    def _on_example_change() -> None:
        chosen = st.session_state.get("selected_example", "Custom input...")
        if chosen in label_to_example:
            st.session_state["original_input"] = label_to_example[chosen]["original"]
            st.session_state["candidate_input"] = label_to_example[chosen]["candidate"]

    st.selectbox(
        "Load a ParaDetox example pair (masked preview)",
        options=option_labels,
        key="selected_example",
        on_change=_on_example_change,
    )

    col1, col2 = st.columns(2)
    with col1:
        original_text = st.text_area(
            "Original comment",
            key="original_input",
            height=140,
            placeholder="Paste or select an original comment...",
        )
    with col2:
        candidate_text = st.text_area(
            "Candidate rewrite",
            key="candidate_input",
            height=140,
            placeholder="Paste or select a candidate rewrite...",
        )

    if st.button("Analyze", type="primary", use_container_width=True):
        if not original_text.strip() or not candidate_text.strip():
            st.warning("Please provide both an original comment and a candidate rewrite.")
        else:
            with st.spinner("Running Toxic-BERT, MiniLM similarity, and Gemini judge..."):
                try:
                    st.session_state["last_report"] = report.build(
                        original_text, candidate_text
                    )
                    st.session_state.pop("rewrite_suggestions", None)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Evaluation failed ({type(exc).__name__}). Please try again.")

    result = st.session_state.get("last_report")
    if result:
        verdict_info = result["verdict"]
        _render_verdict_badge(verdict_info["label"], verdict_info["explanation"])

        tox_before = result["toxicity_before"]
        tox_after = result["toxicity_after"]
        before_val = float(tox_before["toxic"])
        after_val = float(tox_after["toxic"])
        drop_val = float(result["toxicity_drop"])
        sim_score = float(result["similarity"])

        st.subheader("Toxicity Comparison (Toxic-BERT)")
        tcol1, tcol2, tcol3 = st.columns([2, 2, 1])
        with tcol1:
            st.markdown(
                f"**Original toxicity:** `{before_val * 100:.1f}%` "
                f"(top category: `{tox_before['top_category']}`, `{float(tox_before['top_score']) * 100:.1f}%`)"
            )
            st.progress(min(max(before_val, 0.0), 1.0))
        with tcol2:
            st.markdown(
                f"**Candidate toxicity:** `{after_val * 100:.1f}%` "
                f"(top category: `{tox_after['top_category']}`, `{float(tox_after['top_score']) * 100:.1f}%`)"
            )
            st.progress(min(max(after_val, 0.0), 1.0))
        with tcol3:
            st.metric("Toxicity drop", f"{drop_val * 100:.1f}%")

        st.subheader("Meaning Retention & Judge Evidence")
        mcol1, mcol2 = st.columns([1, 2])
        with mcol1:
            st.metric("Meaning retention (MiniLM)", f"{sim_score * 100:.1f}%")
            st.progress(min(max(sim_score, 0.0), 1.0))
            st.caption("Supporting evidence only (not a hard pass/fail gate).")
        with mcol2:
            judge_info = result["judge"]
            status_tag = (
                "Available" if judge_info["available"] else "Unavailable (fallback used)"
            )
            st.markdown(f"**Gemini Judge Status:** `{status_tag}`")
            st.markdown(f"**Meaning preserved:** `{judge_info['meaning_preserved']}`")
            st.markdown(f"**What was lost:** {judge_info['what_was_lost']}")
            st.markdown(f"**Reason:** {judge_info['reason']}")

        st.subheader("Visual Comparison")
        charts_fig = _build_charts_figure(result)
        st.pyplot(charts_fig, use_container_width=True)
        pdf_bytes = _build_pdf_bytes(result, charts_fig)
        plt.close(charts_fig)

        st.subheader("Masked Comment Comparison")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("**Masked original**")
            st.info(result["masked_original"])
        with pcol2:
            st.markdown("**Masked candidate**")
            st.info(result["masked_candidate"])

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                label="Download Report as PDF",
                data=pdf_bytes,
                file_name="rewrite_review_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with dcol2:
            st.download_button(
                label="Download Report as JSON",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name="rewrite_review_report.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Optional: Gemini Rewrite Suggestions")
        if st.button("Suggest Better Rewrites (Gemini)", use_container_width=True):
            with st.spinner("Asking Gemini for non-toxic, meaning-preserving rewrite suggestions..."):
                st.session_state["rewrite_suggestions"] = judge.suggest_rewrites(
                    original_text, candidate_text
                )

        suggestions_state = st.session_state.get("rewrite_suggestions")
        if suggestions_state:
            if suggestions_state["available"] and suggestions_state["suggestions"]:
                for idx, suggestion in enumerate(suggestions_state["suggestions"], start=1):
                    st.success(f"**Option {idx}:** {suggestion}")
            else:
                st.warning(suggestions_state.get("note") or "No suggestions available.")


if __name__ == "__main__":
    main()
