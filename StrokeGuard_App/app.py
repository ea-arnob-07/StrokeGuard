"""StrokeGuard — a polished, multi-page clinical decision support interface."""

from __future__ import annotations

import base64
import html
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import plotly.graph_objects as go
import streamlit as st

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from app_core import (
    APP_NAME,
    APP_SUBTITLE,
    CREDIT_NAME,
    DEFAULT_INPUTS,
    DEMO_INPUTS,
    FEATURE_NAMES,
    SYMPTOM_FIELDS,
    SYMPTOM_GROUPS,
    run_assessment,
)
from pdf_report import generate_pdf_report


ENGINE_PATH = MODULE_DIR / "strokeguard_engine.pkl"
SCALER_PATH = MODULE_DIR / "scaler.pkl"
CSS_PATH = MODULE_DIR / "assets" / "styles.css"

st.set_page_config(
    page_title=f"{APP_NAME} - {APP_SUBTITLE}",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": (
            f"{APP_NAME} — {APP_SUBTITLE}. "
            f"Product design and application engineering by {CREDIT_NAME}."
        )
    },
)


def load_styles() -> None:
    """Load the local visual system."""
    if CSS_PATH.exists():
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[Any, Any]:
    """Load and validate the serialized prediction artifacts."""
    if not ENGINE_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Required assessment files are missing.")

    engine = joblib.load(ENGINE_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected = len(FEATURE_NAMES)
    if (
        getattr(engine, "n_features_in_", None) != expected
        or getattr(scaler, "n_features_in_", None) != expected
    ):
        raise ValueError("Assessment file structure does not match the interface.")
    if not hasattr(engine, "predict_proba"):
        raise TypeError("The assessment service cannot produce an estimate.")
    return engine, scaler


def widget_key(name: str) -> str:
    return f"assessment_{name}"


def set_widget_inputs(values: dict[str, Any]) -> None:
    for name, value in values.items():
        st.session_state[widget_key(name)] = value


def clear_cached_report() -> None:
    for key in ("pdf_report_bytes", "pdf_report_timestamp", "pdf_report_error"):
        st.session_state.pop(key, None)


def query_value(name: str) -> str:
    try:
        value = st.query_params.get(name, "")
    except AttributeError:
        value = ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def render_brand_mark() -> str:
    return f"""
      <div class="sg-brand-lockup">
        <div class="sg-brand-icon" aria-hidden="true">
          <svg viewBox="0 0 48 48" role="img">
            <path class="brain-outline" d="M24 5.5c-5.7 0-9 3.5-9.6 8-4.7.8-7.6 4.4-7.6
              9 0 3.8 2 6.8 5.2 8.4.1 6.6 5 11.8 12 11.8s11.9-5.2
              12-11.8c3.2-1.6 5.2-4.6 5.2-8.4 0-4.6-2.9-8.2-7.6-9-.6-4.5-3.9-8-9.6-8Z"/>
            <path class="brain-lines" d="M24 10v27M16 16c4.3.2 6.8 2.7 8 6M32 16c-4.3.2-6.8
              2.7-8 6M15.5 29c4.5-.1 7.3-2.3 8.5-6M32.5 29c-4.5-.1-7.3-2.3-8.5-6"/>
          </svg>
        </div>
        <div>
          <div class="sg-brand-name">{APP_NAME}</div>
          <div class="sg-brand-subtitle">{APP_SUBTITLE}</div>
        </div>
      </div>
    """


def render_site_bar() -> None:
    st.markdown(
        f"""
        <div class="sg-sitebar">
          {render_brand_mark()}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_notice(compact: bool = False) -> None:
    compact_class = " compact" if compact else ""
    st.markdown(
        f"""
        <div class="sg-decision-notice{compact_class}">
          <div class="sg-notice-icon">+</div>
          <div>
            <b>Professional medical guidance remains essential.</b>
            <p>This application provides an experimental estimate based only on the
            information entered. Do not rely on the result alone for diagnosis,
            treatment, emergency decisions, fitness, or medical clearance. A qualified
            doctor's assessment and advice should always remain the primary basis for care.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_intro(kicker: str, title: str, text: str, step: str = "") -> None:
    step_html = f'<span class="sg-step-chip">{html.escape(step)}</span>' if step else ""
    st.markdown(
        f"""
        <section class="sg-page-intro">
          <div class="sg-eyebrow-row">
            <span class="sg-eyebrow">{html.escape(kicker)}</span>{step_html}
          </div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(text)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <footer class="sg-footer">
          <div class="sg-footer-brand">
            <b>{APP_NAME}</b>
            <span>{APP_SUBTITLE}</span>
          </div>
          <div class="sg-footer-principle">
            <span>Built around a simple principle</span>
            <b>Doctor-guided decisions come first.</b>
          </div>
          <div class="sg-footer-credit">
            <span>Product design & application engineering</span>
            <b>{CREDIT_NAME}</b>
          </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def activate_demo() -> None:
    set_widget_inputs(DEMO_INPUTS)
    st.session_state["assessment_result"] = run_assessment(
        PREDICTION_ENGINE, INPUT_SCALER, DEMO_INPUTS
    )
    st.session_state["preview_initialized"] = True
    clear_cached_report()


def reset_assessment() -> None:
    set_widget_inputs(DEFAULT_INPUTS)
    st.session_state.pop("assessment_result", None)
    st.session_state["preview_initialized"] = True
    clear_cached_report()


def page_home() -> None:
    render_site_bar()

    hero_text, hero_visual = st.columns([1.13, 0.87], gap="large")
    with hero_text:
        st.markdown(
            """
            <div class="sg-hero-copy">
              <div class="sg-live-label"><span></span> GUIDED STROKE AWARENESS</div>
              <h1>Recognize patterns.<br/><em>Understand risk.</em><br/>Act with clarity.</h1>
              <p>StrokeGuard turns a structured symptom review into a clear,
              easy-to-understand assessment experience—helping people prepare for
              a more informed conversation with a qualified doctor.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        primary, secondary, tertiary = st.columns([1.15, 1, 1], gap="small")
        with primary:
            if st.button(
                "Begin symptom assessment",
                type="primary",
                width="stretch",
                key="home_assessment",
            ):
                st.switch_page(ASSESSMENT_PAGE)
        with secondary:
            if st.button(
                "View sample result",
                width="stretch",
                disabled=not SYSTEM_READY,
                key="home_demo",
            ):
                activate_demo()
                st.switch_page(RESULTS_PAGE)
        with tertiary:
            if st.button(
                "Explore stroke signs",
                width="stretch",
                key="home_signs",
            ):
                st.switch_page(SIGNS_PAGE)

        st.markdown(
            """
            <div class="sg-hero-proof">
              <div><b>17</b><span>Health indicators</span></div>
              <div><b>&lt; 2 min</b><span>Guided experience</span></div>
              <div><b>Instant</b><span>Structured summary</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_visual:
        st.markdown(
            """
            <div class="sg-hero-visual" aria-hidden="true">
              <div class="sg-orbit orbit-one"><i></i></div>
              <div class="sg-orbit orbit-two"><i></i></div>
              <div class="sg-visual-glow"></div>
              <div class="sg-brain-shell">
                <svg viewBox="0 0 240 240">
                  <path class="sg-brain-main" d="M120 29c-27 0-43 16-46 38-23 4-38 23-38
                    46 0 19 10 34 26 42 1 34 25 61 58 61s57-27 58-61c16-8
                    26-23 26-42 0-23-15-42-38-46-3-22-19-38-46-38Z"/>
                  <path class="sg-brain-flow" d="M120 51v143M78 81c23 1 36 14 42 32M162
                    81c-23 1-36 14-42 32M76 149c24 0 38-12 44-33M164 149c-24
                    0-38-12-44-33M94 52c2 13 12 21 26 23M146 52c-2 13-12 21-26 23"/>
                </svg>
                <div class="sg-scan-line"></div>
              </div>
              <div class="sg-float-card card-a"><span>01</span><b>Symptoms</b><small>Structured review</small></div>
              <div class="sg-float-card card-b"><span>02</span><b>Insights</b><small>Clear context</small></div>
              <div class="sg-float-card card-c"><span>03</span><b>Next step</b><small>Doctor discussion</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_decision_notice()

    st.markdown(
        """
        <div class="sg-section-title">
          <span>THE STROKEGUARD EXPERIENCE</span>
          <h2>Clarity at every step</h2>
          <p>Designed to make a complex health topic feel structured, calm, and approachable.</p>
        </div>
        <div class="sg-feature-grid">
          <article><i>01</i><div class="sg-feature-symbol">✓</div>
            <h3>Structured symptom review</h3>
            <p>Move through organized indicators without confusing clinical language.</p></article>
          <article><i>02</i><div class="sg-feature-symbol">↗</div>
            <h3>Clear result story</h3>
            <p>See the estimated indicator, assessment range, and selected symptoms together.</p></article>
          <article><i>03</i><div class="sg-feature-symbol">◎</div>
            <h3>Practical next steps</h3>
            <p>Prepare useful questions and topics for a conversation with a doctor.</p></article>
          <article><i>04</i><div class="sg-feature-symbol">↓</div>
            <h3>Portable summary</h3>
            <p>Download a polished PDF that keeps the important context in one place.</p></article>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sg-journey">
          <div>
            <span>YOUR GUIDED JOURNEY</span>
            <h2>From symptoms to a better conversation</h2>
          </div>
          <div class="sg-journey-steps">
            <div><b>1</b><span>Share your current indicators</span></div>
            <i></i>
            <div><b>2</b><span>Review the symptom-based estimate</span></div>
            <i></i>
            <div><b>3</b><span>Discuss the full picture with a doctor</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    about_left, about_action, about_right = st.columns([1.45, 1.1, 1.45])
    with about_action:
        if st.button(
            "Discover the StrokeGuard story",
            width="stretch",
            key="home_about",
        ):
            st.switch_page(ABOUT_PAGE)

    render_footer()


def render_assessment_form() -> dict[str, Any] | None:
    submitted_values: dict[str, Any] | None = None
    form_col, guide_col = st.columns([1.62, 0.78], gap="large")

    with form_col:
        st.markdown(
            """
            <div class="sg-form-heading">
              <div><span>01</span></div>
              <div><h2>Personal details</h2>
              <p>Start with the basic information used in this assessment.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("stroke_assessment_form", border=True):
            demo_age, demo_gender = st.columns([1, 1.35], gap="large")
            with demo_age:
                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=86,
                    step=1,
                    key=widget_key("Age"),
                    help="The currently supported assessment range is age 18 to 86.",
                )
            with demo_gender:
                gender = st.radio(
                    "Sex",
                    options=["Female", "Male"],
                    horizontal=True,
                    key=widget_key("Gender"),
                    help="The current interface supports Female and Male options.",
                )

            st.markdown(
                """
                <div class="sg-form-divider"></div>
                <div class="sg-form-heading inside">
                  <div><span>02</span></div>
                  <div><h2>Current health indicators</h2>
                  <p>Select every item that accurately applies right now.</p></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            group_columns = st.columns(3, gap="medium")
            captured: dict[str, bool] = {}
            group_icons = {"Cardiovascular": "♥", "General": "+", "Other indicators": "◌"}
            for column, (group_name, fields) in zip(
                group_columns, SYMPTOM_GROUPS.items()
            ):
                with column:
                    st.markdown(
                        f"""
                        <div class="sg-symptom-group-title">
                          <span>{group_icons.get(group_name, "+")}</span>
                          {html.escape(group_name)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for field in fields:
                        captured[field] = st.checkbox(
                            dict(SYMPTOM_FIELDS)[field],
                            key=widget_key(field),
                        )

            st.markdown(
                """
                <div class="sg-form-note">
                  Review every selection carefully. The result reflects only the
                  information provided in this form and cannot replace a doctor's
                  history-taking or examination.
                </div>
                """,
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button(
                "Complete my assessment",
                type="primary",
                width="stretch",
                disabled=not SYSTEM_READY,
            )
            if submitted:
                submitted_values = {
                    "Age": int(age),
                    "Gender": gender,
                    **captured,
                }

    with guide_col:
        st.markdown(
            """
            <div class="sg-side-panel">
              <div class="sg-panel-kicker">WHAT HAPPENS NEXT</div>
              <div class="sg-side-step"><span>01</span><div><b>Risk indicator</b>
                <p>A symptom-based estimate shown on a simple 0–100 scale.</p></div></div>
              <div class="sg-side-step"><span>02</span><div><b>Pattern overview</b>
                <p>A clear summary of the indicators you selected.</p></div></div>
              <div class="sg-side-step"><span>03</span><div><b>Discussion guide</b>
                <p>Useful topics to take into a professional consultation.</p></div></div>
              <div class="sg-side-step"><span>04</span><div><b>PDF summary</b>
                <p>A portable record of this assessment.</p></div></div>
            </div>
            <div class="sg-boundary-card">
              <span>ASSESSMENT COVERAGE</span>
              <b>Important context sits outside this form.</b>
              <p>Symptom onset, facial droop, speech or vision changes, medical
              history, medicines, examination findings, laboratory tests, and
              imaging are not included in this estimate.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not SYSTEM_READY:
            st.error("The assessment service is temporarily unavailable.")

    return submitted_values


def page_assessment() -> None:
    render_site_bar()
    render_page_intro(
        "GUIDED SYMPTOM REVIEW",
        "Tell us what you are noticing.",
        "A calm, structured flow that helps organize current indicators before you speak with a doctor.",
        "STEP 1 OF 2",
    )

    sample_col, reset_col, spacer_col = st.columns([0.9, 0.8, 2.3], gap="small")
    with sample_col:
        if st.button(
            "Fill sample profile",
            width="stretch",
            disabled=not SYSTEM_READY,
            key="assessment_demo",
        ):
            set_widget_inputs(DEMO_INPUTS)
            st.session_state.pop("assessment_result", None)
            clear_cached_report()
    with reset_col:
        if st.button("Clear all", width="stretch", key="assessment_clear"):
            reset_assessment()

    submitted = render_assessment_form()
    if submitted is not None and SYSTEM_READY:
        with st.spinner("Preparing your symptom-based assessment…"):
            st.session_state["assessment_result"] = run_assessment(
                PREDICTION_ENGINE, INPUT_SCALER, submitted
            )
            clear_cached_report()
        st.switch_page(RESULTS_PAGE)

    render_decision_notice(compact=True)
    render_footer()


def metric_card(label: str, value: str, detail: str, accent: str = "cyan") -> None:
    st.markdown(
        f"""
        <div class="sg-metric-card accent-{accent}">
          <span>{html.escape(label)}</span>
          <b>{html.escape(value)}</b>
          <small>{html.escape(detail)}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_gauge(result: dict[str, Any]) -> go.Figure:
    score = result["score"]
    color = result["band"]["color"]
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": "%",
                "font": {"size": 48, "color": "#0D1F3C", "family": "Manrope"},
                "valueformat": ".1f",
            },
            title={
                "text": "SYMPTOM-BASED INDICATOR",
                "font": {"size": 12, "color": "#6B7B92", "family": "DM Sans"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 30, 60, 80, 100],
                    "ticktext": ["0", "30", "60", "80", "100"],
                    "tickfont": {"size": 10, "color": "#7B8BA1"},
                },
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "#F1F5FA",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#E6F8F0"},
                    {"range": [30, 60], "color": "#FFF4D7"},
                    {"range": [60, 80], "color": "#FFF0E8"},
                    {"range": [80, 100], "color": "#FDECEF"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.82,
                    "value": score,
                },
            },
        )
    )
    figure.update_layout(
        height=335,
        margin=dict(l=30, r=30, t=55, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans"},
    )
    return figure


def build_sensitivity_chart(result: dict[str, Any]) -> go.Figure | None:
    items = list(reversed(result["sensitivity"][:10]))
    if not items:
        return None
    effects = [item["effect_points"] for item in items]
    labels = [item["feature"] for item in items]
    colors = ["#1AA7A1" if value >= 0 else "#788AA1" for value in effects]
    figure = go.Figure(
        go.Bar(
            x=effects,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{value:+.1f} pts" for value in effects],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Estimated change: %{x:+.2f} points<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color="#C4CFDC")
    figure.update_layout(
        height=max(320, len(items) * 48),
        margin=dict(l=10, r=72, t=20, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "title": "Change in the estimated indicator (percentage points)",
            "gridcolor": "#E9EFF5",
            "zeroline": False,
            "tickfont": {"color": "#738399"},
            "title_font": {"color": "#53647A", "size": 11},
        },
        yaxis={"tickfont": {"color": "#243852", "size": 12}, "automargin": True},
        font={"family": "DM Sans"},
        showlegend=False,
    )
    return figure


def render_result_overview(result: dict[str, Any]) -> None:
    left, right = st.columns([1.04, 0.96], gap="large")
    with left:
        st.plotly_chart(
            build_gauge(result),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        st.caption(
            "This experimental indicator reflects only the information entered. "
            "It is not an absolute medical probability."
        )
    with right:
        band = result["band"]
        st.markdown(
            f"""
            <div class="sg-result-callout" style="--result-color:{band["color"]};
                 --result-soft:{band["soft_color"]};">
              <span>ASSESSMENT RANGE</span>
              <h3>{html.escape(band["label"])}</h3>
              <p>{html.escape(band["description"])}</p>
              <div class="sg-pattern-output">
                <small>Pattern summary</small>
                <b>{html.escape(result["binary_label"])}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        symptoms = result["active_symptoms"]
        pills = (
            "".join(
                f'<span class="sg-active-pill">{html.escape(symptom)}</span>'
                for symptom in symptoms
            )
            if symptoms
            else '<span class="sg-inactive-pill">No symptoms selected</span>'
        )
        st.markdown(
            f"""
            <div class="sg-active-panel">
              <span>SELECTED INDICATORS</span>
              <div class="sg-pill-wrap">{pills}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_indicator_insights(result: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="sg-tab-intro">
          <span>INDICATOR INSIGHTS</span>
          <h3>See how selected information changes the estimate</h3>
          <p>Each selected item is temporarily changed once while all other entries
          stay the same. The chart shows how much the estimated indicator responds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chart = build_sensitivity_chart(result)
    if chart is None:
        st.info(
            "Select one or more symptoms, or use an age different from the reference, "
            "to view indicator insights."
        )
    else:
        st.plotly_chart(
            chart,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
    st.markdown(
        """
        <div class="sg-method-note">
          <b>How to read this:</b> A positive value means the current entry increased
          the estimate compared with the displayed alternative. It does not prove
          cause, severity, or an individual medical contribution.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_next_steps(result: dict[str, Any]) -> None:
    guidance_col, discussion_col = st.columns([1.03, 0.97], gap="large")
    with guidance_col:
        st.markdown(
            """
            <div class="sg-tab-intro compact">
              <span>NEXT STEPS</span><h3>Practical, input-aware guidance</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in result["guidance"]:
            st.markdown(
                f"""
                <div class="sg-guidance-card tone-{item["tone"]}">
                  <b>{html.escape(item["title"])}</b>
                  <p>{html.escape(item["text"])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with discussion_col:
        st.markdown(
            """
            <div class="sg-tab-intro compact">
              <span>DOCTOR DISCUSSION</span><h3>Topics to bring to an appointment</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for index, item in enumerate(result["clinical_discussions"], 1):
            st.markdown(
                f"""
                <div class="sg-discussion-row">
                  <span>{index:02d}</span>
                  <div><b>{html.escape(item["name"])}</b>
                  <p>{html.escape(item["reason"])}</p></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div class="sg-method-note">
              A qualified doctor should decide whether any examination or test is
              appropriate after reviewing onset, history, medicines, and physical findings.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_report(result: dict[str, Any]) -> None:
    report_col, coverage_col = st.columns([1, 1], gap="large")
    with report_col:
        st.markdown(
            """
            <div class="sg-tab-intro compact">
              <span>PORTABLE SUMMARY</span><h3>Download your assessment PDF</h3>
              <p>Keep the recorded inputs, estimate, indicator insights, discussion
              topics, and responsible-use note together.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        report_timestamp = result["assessed_at"]
        if st.session_state.get("pdf_report_timestamp") != report_timestamp:
            try:
                st.session_state["pdf_report_bytes"] = generate_pdf_report(result)
                st.session_state["pdf_report_timestamp"] = report_timestamp
                st.session_state.pop("pdf_report_error", None)
            except Exception as exc:
                st.session_state["pdf_report_bytes"] = None
                st.session_state["pdf_report_error"] = str(exc)

        pdf_bytes = st.session_state.get("pdf_report_bytes")
        if pdf_bytes:
            filename_time = datetime.fromisoformat(result["assessed_at"]).strftime(
                "%Y%m%d_%H%M%S"
            )
            st.download_button(
                "Download assessment summary",
                data=pdf_bytes,
                file_name=f"StrokeGuard_Assessment_{filename_time}.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
            )
            st.caption(f"Report experience created by {CREDIT_NAME}")
        else:
            st.error("The PDF summary could not be prepared. Please try again.")

    with coverage_col:
        st.markdown(
            """
            <div class="sg-tab-intro compact">
              <span>WHAT THIS RESULT CONSIDERS</span><h3>A focused snapshot</h3>
            </div>
            <div class="sg-coverage-grid">
              <div><span>Input scope</span><b>Age, sex, and 15 indicators</b></div>
              <div><span>Result format</span><b>Estimate and assessment range</b></div>
              <div><span>Context</span><b>Information entered in this session</b></div>
              <div><span>Professional review</span><b>Doctor interpretation remains essential</b></div>
            </div>
            <div class="sg-method-note">
              The estimate cannot see sudden symptom onset, examination findings,
              laboratory results, imaging, complete history, or medicines.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_results_workspace(result: dict[str, Any]) -> None:
    assessed_time = datetime.fromisoformat(result["assessed_at"]).strftime("%I:%M %p")
    metrics = st.columns(4, gap="medium")
    with metrics[0]:
        metric_card("Risk indicator", f'{result["score"]:.1f}%', "Symptom-based estimate", "cyan")
    with metrics[1]:
        metric_card("Assessment range", result["band"]["short_label"], "Contextual display band", "amber")
    with metrics[2]:
        metric_card(
            "Selected symptoms",
            str(len(result["active_symptoms"])),
            f"of {len(SYMPTOM_FIELDS)} indicators",
            "violet",
        )
    with metrics[3]:
        metric_card("Completed", assessed_time, "Your local time", "slate")

    if result["score"] >= 80:
        st.markdown(
            """
            <div class="sg-urgent-banner">
              <b>Very high symptom-based indicator</b>
              <p>Please arrange prompt professional review. If sudden facial droop,
              arm weakness, speech or vision changes, severe balance loss, or another
              acute neurological sign is present, seek emergency care immediately.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    overview_tab, insights_tab, next_tab, report_tab = st.tabs(
        ["Overview", "Indicator insights", "Next steps", "Download summary"]
    )
    with overview_tab:
        render_result_overview(result)
    with insights_tab:
        render_indicator_insights(result)
    with next_tab:
        render_next_steps(result)
    with report_tab:
        render_report(result)


def page_results() -> None:
    render_site_bar()
    render_page_intro(
        "YOUR STROKEGUARD SUMMARY",
        "A clearer view of the information you entered.",
        "Review the estimate, understand the selected indicators, and prepare for a doctor-guided next step.",
        "STEP 2 OF 2",
    )

    result = st.session_state.get("assessment_result")
    if not result:
        st.markdown(
            """
            <div class="sg-empty-state">
              <div class="sg-empty-orbit"><span>SG</span></div>
              <div>
                <span>NO ASSESSMENT YET</span>
                <h3>Your result workspace is ready when you are.</h3>
                <p>Complete the guided symptom review or open the sample experience.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        action_a, action_b, action_space = st.columns([1, 1, 2], gap="small")
        with action_a:
            if st.button(
                "Start assessment",
                type="primary",
                width="stretch",
                key="results_start",
            ):
                st.switch_page(ASSESSMENT_PAGE)
        with action_b:
            if st.button(
                "Open sample",
                width="stretch",
                disabled=not SYSTEM_READY,
                key="results_demo",
            ):
                activate_demo()
                st.switch_page(RESULTS_PAGE)
    else:
        render_results_workspace(result)
        render_decision_notice(compact=True)
        action_a, action_b, action_space = st.columns([1, 1, 2], gap="small")
        with action_a:
            if st.button(
                "Start a new assessment",
                type="primary",
                width="stretch",
                key="results_new",
            ):
                reset_assessment()
                st.switch_page(ASSESSMENT_PAGE)
        with action_b:
            if st.button(
                "Review stroke signs",
                width="stretch",
                key="results_signs",
            ):
                st.switch_page(SIGNS_PAGE)

    render_footer()


def page_stroke_signs() -> None:
    render_site_bar()
    render_page_intro(
        "RECOGNIZE THE WARNING SIGNS",
        "Know B.E. F.A.S.T. Every minute matters.",
        "Sudden neurological changes can be a medical emergency. Recognizing the pattern and acting quickly can make a critical difference.",
    )

    st.markdown(
        """
        <div class="sg-befast-grid">
          <article class="b"><strong>B</strong><span>BALANCE</span><h3>Sudden loss of balance</h3>
            <p>Watch for new dizziness, poor coordination, or trouble walking.</p></article>
          <article class="e"><strong>E</strong><span>EYES</span><h3>Sudden vision change</h3>
            <p>Blurred, double, or lost vision in one or both eyes may be a warning sign.</p></article>
          <article class="f"><strong>F</strong><span>FACE</span><h3>One side drooping</h3>
            <p>Ask the person to smile. Check whether the smile looks uneven.</p></article>
          <article class="a"><strong>A</strong><span>ARMS</span><h3>Arm weakness or numbness</h3>
            <p>Ask the person to raise both arms. One arm may drift downward.</p></article>
          <article class="s"><strong>S</strong><span>SPEECH</span><h3>Speech difficulty</h3>
            <p>Speech may be slurred, strange, or difficult to understand.</p></article>
          <article class="t"><strong>T</strong><span>TIME</span><h3>Act immediately</h3>
            <p>Contact local emergency services. Note when the symptoms first appeared.</p></article>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sg-emergency-callout">
          <div>!</div>
          <section>
            <span>DO NOT WAIT FOR THE APP RESULT</span>
            <h2>If any warning sign appears suddenly, seek emergency care now.</h2>
            <p>Symptoms that improve or disappear can still require urgent evaluation.
            This assessment must never delay emergency action.</p>
          </section>
        </div>
        """,
        unsafe_allow_html=True,
    )

    factors_col, habits_col = st.columns(2, gap="large")
    with factors_col:
        st.markdown(
            """
            <div class="sg-info-panel risk">
              <span>KNOW YOUR RISK FACTORS</span>
              <h2>Some risks can be managed</h2>
              <div class="sg-chip-grid">
                <b>High blood pressure</b><b>Tobacco use</b>
                <b>High cholesterol</b><b>Diabetes</b>
                <b>Physical inactivity</b><b>Unhealthy diet</b>
                <b>Excess body weight</b><b>Irregular heart rhythm</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with habits_col:
        st.markdown(
            """
            <div class="sg-info-panel habits">
              <span>SUPPORT VASCULAR HEALTH</span>
              <h2>Small habits, sustained over time</h2>
              <ul>
                <li>Follow a doctor's plan for blood pressure and related conditions.</li>
                <li>Avoid tobacco and reduce exposure to second-hand smoke.</li>
                <li>Choose balanced meals and moderate salt, saturated fat, and sugar.</li>
                <li>Stay physically active in a way approved for your health.</li>
                <li>Take prescribed medicines exactly as directed.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="sg-source-bar">
          <span>Health information references</span>
          <a href="https://www.stroke.org/en/about-stroke/stroke-symptoms" target="_blank">
            American Stroke Association ↗</a>
          <a href="https://www.cdc.gov/stroke/signs-symptoms/index.html" target="_blank">
            CDC Stroke Signs ↗</a>
          <a href="https://www.who.int/news-room/fact-sheets/detail/stroke" target="_blank">
            World Health Organization ↗</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer()


def page_about() -> None:
    render_site_bar()
    render_page_intro(
        "ABOUT STROKEGUARD",
        "Health information should feel clear, calm, and responsible.",
        "StrokeGuard is a guided symptom assessment experience created to organize information and support a better conversation with a qualified doctor.",
    )

    mission_col, principles_col = st.columns([1.05, 0.95], gap="large")
    with mission_col:
        st.markdown(
            """
            <div class="sg-about-card large">
              <span>OUR PURPOSE</span>
              <h2>Turn scattered symptoms into a structured overview.</h2>
              <p>People often struggle to explain what they are feeling. StrokeGuard
              brings age, sex, and selected health indicators into one consistent
              flow, then presents the result in clear language that is easier to
              take into a professional consultation.</p>
              <div class="sg-about-stats">
                <div><b>17</b><span>Structured inputs</span></div>
                <div><b>5</b><span>Focused pages</span></div>
                <div><b>1</b><span>Portable summary</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with principles_col:
        st.markdown(
            """
            <div class="sg-principles">
              <span>EXPERIENCE PRINCIPLES</span>
              <div><i>01</i><section><b>Clarity first</b>
                <p>Simple explanations and a predictable guided flow.</p></section></div>
              <div><i>02</i><section><b>Context over certainty</b>
                <p>The estimate is always presented with its limitations.</p></section></div>
              <div><i>03</i><section><b>Doctor-guided care</b>
                <p>Professional assessment remains the basis for decisions.</p></section></div>
              <div><i>04</i><section><b>Emergency awareness</b>
                <p>Sudden warning signs must never wait for an app result.</p></section></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_decision_notice()

    try:
        img_path = MODULE_DIR / "assets" / "IMG_3934 (2).JPG"
        with open(img_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        monogram_html = f'<img src="data:image/jpeg;base64,{img_b64}" class="sg-credit-monogram" style="object-fit: cover;" />'
    except Exception:
        monogram_html = '<div class="sg-credit-monogram">EAA</div>'

    st.markdown(
        f"""
        <div class="sg-credit-profile">
          {monogram_html}
          <div>
            <span>PRODUCT DESIGN & APPLICATION ENGINEERING</span>
            <h2>{CREDIT_NAME}</h2>
            <p>Responsible for the StrokeGuard brand experience, interface architecture,
            interaction design, assessment workflow, result presentation, and report system.</p>
          </div>
          <div class="sg-credit-seal">
            <span>CREATED WITH</span><b>Clarity · Care · Craft</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer()


load_styles()

try:
    PREDICTION_ENGINE, INPUT_SCALER = load_artifacts()
    SYSTEM_READY = True
    SYSTEM_ERROR = None
except Exception as exc:
    PREDICTION_ENGINE = INPUT_SCALER = None
    SYSTEM_READY = False
    SYSTEM_ERROR = str(exc)

preview_mode = query_value("demo") == "1"
if "inputs_initialized" not in st.session_state:
    set_widget_inputs(DEMO_INPUTS if preview_mode else DEFAULT_INPUTS)
    st.session_state["inputs_initialized"] = True
    st.session_state["preview_initialized"] = False

if preview_mode and SYSTEM_READY and not st.session_state.get("preview_initialized"):
    activate_demo()

HOME_PAGE = st.Page(
    page_home,
    title="Home",
    icon=":material/home:",
    url_path="home",
    default=True,
)
ASSESSMENT_PAGE = st.Page(
    page_assessment,
    title="Assessment",
    icon=":material/assignment:",
    url_path="assessment",
)
RESULTS_PAGE = st.Page(
    page_results,
    title="Results",
    icon=":material/monitoring:",
    url_path="results",
)
SIGNS_PAGE = st.Page(
    page_stroke_signs,
    title="Stroke signs",
    icon=":material/emergency:",
    url_path="stroke-signs",
)
ABOUT_PAGE = st.Page(
    page_about,
    title="About",
    icon=":material/info:",
    url_path="about",
)

navigation = st.navigation(
    [HOME_PAGE, ASSESSMENT_PAGE, RESULTS_PAGE, SIGNS_PAGE, ABOUT_PAGE],
    position="top",
)
navigation.run()
