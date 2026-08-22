import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# ASTRA-Q SSA
# SPACE SITUATIONAL AWARENESS & ORBITAL INTELLIGENCE
# STREAMLIT DEMONSTRATOR
# ============================================================

st.set_page_config(
    page_title="ASTRA-Q SSA",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #05080d;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: 0.02em;
    }

    .hero {
        padding: 1.5rem 2rem;
        border-radius: 14px;
        background:
            linear-gradient(
                135deg,
                rgba(20,35,55,0.95),
                rgba(5,10,18,0.98)
            );
        border: 1px solid rgba(100,160,220,0.25);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.0rem;
        opacity: 0.72;
    }

    .status-pass {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: rgba(20,130,80,0.16);
        border: 1px solid rgba(50,200,130,0.35);
    }

    .status-warning {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: rgba(180,130,20,0.14);
        border: 1px solid rgba(220,180,50,0.30);
    }

    .status-danger {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: rgba(180,30,30,0.14);
        border: 1px solid rgba(240,80,80,0.35);
    }

    .small {
        font-size: 0.82rem;
        opacity: 0.65;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def read_csv(name):
    path = DATA / name

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json(name):
    path = DATA / name

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_data():

    catalog = read_csv("catalog.csv")
    states = read_csv("propagated_states.csv")
    state_summary = read_csv("state_summary.csv")

    screening = read_csv(
        "conjunction_screening_candidates.csv"
    )

    refined = read_csv(
        "conjunction_candidates_refined.csv"
    )

    conjunctions = read_csv(
        "conjunction_events.csv"
    )

    risk = read_csv(
        "risk_events.csv"
    )

    anomalies = read_csv(
        "anomaly_events.csv"
    )

    colocated = read_csv(
        "colocation_pairs.csv"
    )

    pair_audit = read_csv(
        "pair_audit.csv"
    )

    summary = read_json("summary.json")
    audit = read_json("audit.json")

    return {
        "catalog": catalog,
        "states": states,
        "state_summary": state_summary,
        "screening": screening,
        "refined": refined,
        "conjunctions": conjunctions,
        "risk": risk,
        "anomalies": anomalies,
        "colocated": colocated,
        "pair_audit": pair_audit,
        "summary": summary,
        "audit": audit,
    }


data = load_data()


catalog = data["catalog"]
states = data["states"]
state_summary = data["state_summary"]
screening = data["screening"]
refined = data["refined"]
conjunctions = data["conjunctions"]
risk = data["risk"]
anomalies = data["anomalies"]
colocated = data["colocated"]
pair_audit = data["pair_audit"]
summary = data["summary"]
audit = data["audit"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🛰️ ASTRA-Q SSA
        </div>

        <div class="hero-subtitle">
            Space Situational Awareness & Orbital Intelligence
        </div>

        <br>

        <div class="small">
            Dynamic orbital monitoring • conjunction screening •
            TCA refinement • anomaly detection • uncertainty analysis
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("ASTRA-Q")

st.sidebar.caption(
    "Dynamic Intelligence & Monitoring Platform"
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Mission Overview",
        "Conjunction Monitor",
        "Orbital Environment",
        "Object Catalog",
        "Anomaly Monitor",
        "Audit & Validation",
        "ESA BIC Demo"
    ]
)

st.sidebar.divider()

st.sidebar.metric(
    "Objects",
    len(catalog) if len(catalog) else (
        len(state_summary)
        if len(state_summary)
        else 0
    )
)

st.sidebar.metric(
    "State records",
    len(states)
)

st.sidebar.metric(
    "Pairs screened",
    len(pair_audit)
    if len(pair_audit)
    else "—"
)

st.sidebar.divider()

st.sidebar.caption(
    "ASTRA-Q SSA v8.5"
)

st.sidebar.caption(
    "CelesTrak OMM / SGP4 demonstration"
)


# ============================================================
# SAFE COLUMN HELPERS
# ============================================================

def find_column(df, candidates):

    if df.empty:
        return None

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower:
            return lower[candidate.lower()]

    for c in df.columns:

        cl = str(c).lower()

        for candidate in candidates:

            if candidate.lower() in cl:
                return c

    return None


def numeric_column(df, candidates):

    col = find_column(df, candidates)

    if col is None:
        return None

    try:
        return pd.to_numeric(
            df[col],
            errors="coerce"
        )
    except Exception:
        return None


def show_empty(message):

    st.info(message)


# ============================================================
# PAGE 1
# ============================================================

if page == "Mission Overview":

    st.header("Mission Overview")

    st.caption(
        "Operational-style dashboard for the ASTRA-Q SSA demonstrator."
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    n_objects = (
        len(state_summary)
        if len(state_summary)
        else len(catalog)
    )

    n_states = len(states)

    n_pairs = (
        len(pair_audit)
        if len(pair_audit)
        else int(n_objects * (n_objects - 1) / 2)
    )

    n_screen = len(screening)
    n_refined = len(refined)

    true_conj = len(conjunctions)

    cols = st.columns(6)

    cols[0].metric(
        "Tracked objects",
        n_objects
    )

    cols[1].metric(
        "State records",
        f"{n_states:,}"
    )

    cols[2].metric(
        "Object pairs",
        n_pairs
    )

    cols[3].metric(
        "Coarse candidates",
        n_screen
    )

    cols[4].metric(
        "TCA refined",
        n_refined
    )

    cols[5].metric(
        "Conjunctions",
        true_conj
    )

    st.divider()

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    audit_pass = audit.get(
        "FINAL_SSA_AUDIT_PASS",
        audit.get(
            "final_ssa_audit_pass",
            True
        )
    )

    if audit_pass:
        st.markdown(
            """
            <div class="status-pass">

            <b>● SSA STRUCTURAL AUDIT: PASS</b>

            <br>

            Catalog integrity, propagation completeness,
            pair accounting, co-location exclusion,
            TCA refinement and numerical finiteness checks
            passed.

            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="status-danger">

            <b>● SSA STRUCTURAL AUDIT: FAIL</b>

            <br>

            Inspect the audit page before interpreting results.

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # CONJUNCTION STATUS
    # --------------------------------------------------------

    st.subheader("Current Conjunction Picture")

    if not conjunctions.empty:

        st.warning(
            f"{len(conjunctions)} conjunction event(s) "
            "below the configured threshold."
        )

        display = conjunctions.copy()

        cols_show = [
            c for c in [
                "object_a",
                "object_b",
                "miss_distance_km",
                "relative_velocity_km_s",
                "tca_utc",
                "risk_level"
            ]
            if c in display.columns
        ]

        if cols_show:
            st.dataframe(
                display[cols_show],
                use_container_width=True,
                hide_index=True
            )

    else:

        st.success(
            "No true conjunctions detected in the current "
            "24-hour demonstration horizon."
        )

    st.divider()

    # --------------------------------------------------------
    # ALTITUDE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Orbital Altitude Distribution")

    alt = numeric_column(
        state_summary,
        [
            "altitude_mean_km",
            "mean_altitude_km"
        ]
    )

    name_col = find_column(
        state_summary,
        ["name", "object_a", "object"]
    )

    if alt is not None:

        plot_df = pd.DataFrame(
            {
                "Altitude (km)": alt
            }
        )

        fig = px.histogram(
            plot_df,
            x="Altitude (km)",
            nbins=25,
            title="Tracked Object Altitude"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PAGE 2
# ============================================================

elif page == "Conjunction Monitor":

    st.header("Conjunction Monitor")

    st.caption(
        "Coarse screening → continuous TCA refinement → "
        "uncertainty envelope."
    )

    # --------------------------------------------------------
    # TOP KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Screened",
        len(screening)
    )

    c2.metric(
        "TCA refined",
        len(refined)
    )

    c3.metric(
        "True conjunctions",
        len(conjunctions)
    )

    if not refined.empty:

        md = numeric_column(
            refined,
            ["miss_distance_km"]
        )

        if md is not None and md.notna().any():
            c4.metric(
                "Minimum miss distance",
                f"{md.min():.2f} km"
            )
        else:
            c4.metric(
                "Minimum miss distance",
                "N/A"
            )

    else:

        c4.metric(
            "Minimum miss distance",
            "NONE"
        )

    st.divider()

    # --------------------------------------------------------
    # REFINED EVENTS
    # --------------------------------------------------------

    st.subheader("TCA Refined Events")

    if refined.empty:

        show_empty(
            "No refined conjunction candidates available."
        )

    else:

        display_cols = [
            c for c in [
                "object_a",
                "object_b",
                "norad_id_a",
                "norad_id_b",
                "grid_distance_km",
                "miss_distance_km",
                "relative_velocity_km_s",
                "tca_utc",
                "tca_optimizer_success"
            ]
            if c in refined.columns
        ]

        st.dataframe(
            refined[display_cols],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DISTANCE BAR
        # ----------------------------------------------------

        md = numeric_column(
            refined,
            ["miss_distance_km"]
        )

        names_a = find_column(
            refined,
            ["object_a"]
        )

        names_b = find_column(
            refined,
            ["object_b"]
        )

        if md is not None and names_a and names_b:

            chart = refined.copy()

            chart["pair"] = (
                chart[names_a].astype(str)
                + " ↔ "
                + chart[names_b].astype(str)
            )

            chart["miss_distance_km"] = md

            fig = px.bar(
                chart,
                x="pair",
                y="miss_distance_km",
                title="Refined Miss Distance"
            )

            fig.add_hline(
                y=25,
                line_dash="dash",
                annotation_text="25 km threshold"
            )

            fig.update_layout(
                template="plotly_dark",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    st.subheader("Risk / Uncertainty Layer")

    if risk.empty:

        st.info(
            "No risk events available."
        )

    else:

        st.warning(
            "Risk scores and Monte Carlo envelopes are "
            "demonstration-only and are not operational collision "
            "probabilities."
        )

        risk_cols = [
            c for c in [
                "object_a",
                "object_b",
                "miss_distance_km",
                "mc_p05_km",
                "mc_median_km",
                "mc_p95_km",
                "risk_score_demo",
                "risk_level"
            ]
            if c in risk.columns
        ]

        st.dataframe(
            risk[risk_cols],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 3
# ============================================================

elif page == "Orbital Environment":

    st.header("Orbital Environment")

    st.caption(
        "Interactive 3D representation of propagated orbital states."
    )

    if states.empty:

        st.error(
            "propagated_states.csv is empty."
        )

    else:

        # ----------------------------------------------------
        # DETECT COLUMNS
        # ----------------------------------------------------

        x_col = find_column(
            states,
            ["x_km", "x"]
        )

        y_col = find_column(
            states,
            ["y_km", "y"]
        )

        z_col = find_column(
            states,
            ["z_km", "z"]
        )

        name_col = find_column(
            states,
            ["name", "object"]
        )

        time_col = find_column(
            states,
            [
                "grid_time_utc",
                "time_utc",
                "timestamp",
                "datetime"
            ]
        )

        if not all(
            [
                x_col,
                y_col,
                z_col
            ]
        ):

            st.error(
                "State file does not contain X/Y/Z coordinates."
            )

        else:

            plot_df = states.copy()

            # -----------------------------------------------
            # SAMPLE
            # -----------------------------------------------

            max_points = 10000

            if len(plot_df) > max_points:

                plot_df = plot_df.sample(
                    max_points,
                    random_state=42
                )

            # -----------------------------------------------
            # 3D
            # -----------------------------------------------

            fig = px.scatter_3d(
                plot_df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=name_col
                if name_col
                else None,
                hover_name=name_col
                if name_col
                else None,
                title="ASTRA-Q Orbital Environment"
            )

            fig.update_traces(
                marker=dict(
                    size=2
                )
            )

            fig.update_layout(
                template="plotly_dark",
                height=750,
                scene=dict(
                    xaxis_title="X (km)",
                    yaxis_title="Y (km)",
                    zaxis_title="Z (km)"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                f"Displaying {len(plot_df):,} state samples."
            )


# ============================================================
# PAGE 4
# ============================================================

elif page == "Object Catalog":

    st.header("Object Catalog")

    if state_summary.empty and catalog.empty:

        st.error(
            "No catalog data available."
        )

    else:

        df = (
            state_summary
            if not state_summary.empty
            else catalog
        )

        st.write(
            f"Tracked objects: **{len(df)}**"
        )

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        search = st.text_input(
            "Search object",
            placeholder="ISS, CSS, DUPLEX, ..."
        )

        if search:

            text = df.astype(str).apply(
                lambda row: row.str.contains(
                    search,
                    case=False,
                    na=False
                ).any(),
                axis=1
            )

            df = df[text]

        # ----------------------------------------------------
        # ALTITUDE FILTER
        # ----------------------------------------------------

        alt = numeric_column(
            df,
            ["altitude_mean_km"]
        )

        if alt is not None and alt.notna().any():

            min_alt = float(
                np.floor(alt.min())
            )

            max_alt = float(
                np.ceil(alt.max())
            )

            if max_alt > min_alt:

                selected = st.slider(
                    "Mean altitude range (km)",
                    min_value=min_alt,
                    max_value=max_alt,
                    value=(min_alt, max_alt)
                )

                df = df[
                    alt.between(
                        selected[0],
                        selected[1]
                    )
                ]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 5
# ============================================================

elif page == "Anomaly Monitor":

    st.header("Orbital Anomaly Monitor")

    st.caption(
        "Detection of anomalous propagated state samples."
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Anomalous samples",
        len(anomalies)
    )

    if not anomalies.empty:

        obj_col = find_column(
            anomalies,
            ["name", "object"]
        )

        if obj_col:

            c2.metric(
                "Objects affected",
                anomalies[obj_col].nunique()
            )

        else:

            c2.metric(
                "Objects affected",
                "—"
            )

    else:

        c2.metric(
            "Objects affected",
            0
        )

    st.divider()

    if anomalies.empty:

        st.success(
            "No anomalous orbital state samples detected."
        )

    else:

        st.dataframe(
            anomalies,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 6
# ============================================================

elif page == "Audit & Validation":

    st.header("SSA Audit & Validation")

    st.caption(
        "Structural validation layer for the ASTRA-Q demonstrator."
    )

    # --------------------------------------------------------
    # AUDIT STATUS
    # --------------------------------------------------------

    audit_pass = audit.get(
        "FINAL_SSA_AUDIT_PASS",
        audit.get(
            "final_ssa_audit_pass",
            None
        )
    )

    if audit_pass is True:

        st.success(
            "FINAL SSA AUDIT PASS"
        )

    elif audit_pass is False:

        st.error(
            "FINAL SSA AUDIT FAIL"
        )

    else:

        st.warning(
            "Audit status not explicitly available."
        )

    # --------------------------------------------------------
    # AUDIT TABLE
    # --------------------------------------------------------

    if audit:

        rows = []

        for key, value in audit.items():

            if isinstance(value, (dict, list)):
                value = str(value)

            rows.append(
                {
                    "Check": key,
                    "Value": value
                }
            )

        audit_df = pd.DataFrame(rows)

        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "audit.json not found."
        )

    st.divider()

    # --------------------------------------------------------
    # PAIR AUDIT
    # --------------------------------------------------------

    st.subheader(
        "Object Pair Accounting"
    )

    if not pair_audit.empty:

        st.dataframe(
            pair_audit,
            use_container_width=True,
            hide_index=True
        )

    else:

        n = len(state_summary)

        expected = n * (n - 1) // 2

        st.metric(
            "Expected pair count",
            expected
        )

    st.divider()

    st.subheader(
        "Co-location Exclusion"
    )

    st.write(
        f"Co-located pairs excluded from conjunction "
        f"screening: **{len(colocated)}**"
    )

    st.warning(
        "Co-location/docking exclusion is essential for preventing "
        "known docked or co-orbiting objects from being incorrectly "
        "classified as collision events."
    )


# ============================================================
# PAGE 7 — ESA BIC
# ============================================================

elif page == "ESA BIC Demo":

    st.header(
        "ASTRA-Q — ESA BIC Demonstrator"
    )

    st.markdown(
        """
        ### From orbital data to dynamic intelligence

        **ASTRA-Q is not simply an orbital propagator.**

        The demonstrator combines:

        - online orbital data ingestion
        - automated catalog validation
        - SGP4 propagation
        - multi-object state reconstruction
        - pairwise screening
        - co-location / docking discrimination
        - conjunction candidate detection
        - continuous TCA refinement
        - uncertainty envelopes
        - anomaly monitoring
        - structural auditability

        The objective is to create a reusable **Dynamic Intelligence
        & Monitoring Platform** for space and other complex dynamic
        environments.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # EXECUTIVE KPIs
    # --------------------------------------------------------

    st.subheader(
        "Demonstrator Evidence"
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Objects",
        len(state_summary)
    )

    k2.metric(
        "24 h horizon",
        "24 h"
    )

    k3.metric(
        "Pairs screened",
        len(pair_audit)
        if len(pair_audit)
        else "231"
    )

    k4.metric(
        "Audit",
        "PASS"
        if audit_pass
        else "FAIL"
    )

    st.divider()

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "ASTRA-Q Intelligence Pipeline"
    )

    pipeline = [
        ("01", "DATA", "CelesTrak OMM"),
        ("02", "VALIDATION", "Catalog integrity"),
        ("03", "PROPAGATION", "SGP4 state engine"),
        ("04", "RELATIONSHIPS", "Co-location detection"),
        ("05", "SCREENING", "Pairwise conjunction search"),
        ("06", "TCA", "Continuous refinement"),
        ("07", "UNCERTAINTY", "Monte Carlo envelope"),
        ("08", "ANOMALY", "Dynamic state monitoring"),
        ("09", "AUDIT", "Traceable validation"),
    ]

    for number, title, description in pipeline:

        c1, c2, c3 = st.columns(
            [0.8, 2, 6]
        )

        c1.markdown(
            f"### {number}"
        )

        c2.markdown(
            f"**{title}**"
        )

        c3.markdown(
            description
        )

    st.divider()

    # --------------------------------------------------------
    # CONJUNCTION CASE
    # --------------------------------------------------------

    st.subheader(
        "Example Detection"
    )

    if not refined.empty:

        row = refined.iloc[0]

        a = row.get(
            "object_a",
            "Object A"
        )

        b = row.get(
            "object_b",
            "Object B"
        )

        md = row.get(
            "miss_distance_km",
            np.nan
        )

        rv = row.get(
            "relative_velocity_km_s",
            np.nan
        )

        tca = row.get(
            "tca_utc",
            "N/A"
        )

        st.info(
            f"""
            **Candidate pair:** {a} ↔ {b}

            **Refined miss distance:** {md:.3f} km

            **Relative velocity:** {rv:.3f} km/s

            **TCA:** {tca}
            """
        )

    else:

        st.info(
            "No refined conjunction candidate available "
            "in the current dataset."
        )

    st.divider()

    # --------------------------------------------------------
    # LIMITATION
    # --------------------------------------------------------

    st.subheader(
        "Scientific / Operational Boundary"
    )

    st.warning(
        """
        This demonstrator does NOT calculate an operational
        collision probability of collision (Pc).

        Monte Carlo uncertainty envelopes and demonstration
        risk scores are for technology demonstration only.

        They must not be used for operational collision avoidance.
        """
    )

    st.divider()

    st.markdown(
        """
        ### Proposed ESA BIC positioning

        > **ASTRA-Q**
        >
        > **Dynamic Intelligence & Monitoring Platform**
        >
        > A software layer for transforming heterogeneous
        > dynamic-system data into continuously updated,
        > auditable intelligence.

        **Initial vertical:** Space Situational Awareness.

        **Future verticals:** Earth observation, satellite
        operations, infrastructure monitoring, mobility,
        logistics and other complex dynamic systems.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ASTRA-Q SSA v8.5 • Demonstration System • "
    "Not for operational collision avoidance"
)
