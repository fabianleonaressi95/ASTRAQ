import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from astra_q_ssa import (
    VERSION,
    export_results,
    run_ssa,
)


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="ASTRA-Q SSA",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .status {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ ASTRA-Q SSA</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Space Situational Awareness & Orbital Intelligence Platform"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("ASTRA-Q CONTROL")

max_objects = st.sidebar.slider(
    "Maximum objects",
    5,
    40,
    22,
)

horizon = st.sidebar.selectbox(
    "Propagation horizon",
    [6, 12, 24, 48],
    index=2,
)

step = st.sidebar.selectbox(
    "Propagation step",
    [1, 5, 10],
    index=1,
)

screening = st.sidebar.number_input(
    "Screening distance [km]",
    min_value=10.0,
    max_value=500.0,
    value=50.0,
)

threshold = st.sidebar.number_input(
    "Conjunction threshold [km]",
    min_value=1.0,
    max_value=100.0,
    value=25.0,
)

run_button = st.sidebar.button(
    "🚀 RUN SSA ANALYSIS",
    type="primary",
    use_container_width=True,
)


# ============================================================
# SESSION
# ============================================================

if "results" not in st.session_state:

    st.session_state.results = None


# ============================================================
# RUN
# ============================================================

if run_button:

    with st.spinner(
        "Running ASTRA-Q orbital intelligence engine..."
    ):

        try:

            results = run_ssa(
                max_objects=max_objects,
                horizon_hours=horizon,
                step_minutes=step,
                screening_distance_km=screening,
                conjunction_threshold_km=threshold,
            )

            st.session_state.results = results

            export_results(
                results
            )

            st.success(
                "ASTRA-Q SSA analysis completed."
            )

        except Exception as exc:

            st.error(
                "SSA execution failed."
            )

            st.exception(exc)


# ============================================================
# NO RESULT
# ============================================================

if st.session_state.results is None:

    st.info(
        """
        ### ASTRA-Q SSA READY

        Press **RUN SSA ANALYSIS** to start the orbital
        intelligence pipeline.

        The engine will:

        1. acquire orbital data
        2. validate the catalog
        3. construct SGP4 objects
        4. propagate the constellation
        5. detect co-located objects
        6. screen conjunctions
        7. refine TCA
        8. calculate demonstration risk
        9. detect orbital anomalies
        10. execute structural audit
        """
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

r = st.session_state.results

summary = r["summary"]


# ============================================================
# STATUS
# ============================================================

source = summary["source"]

if source == "LIVE_CELESTRAK":

    st.success(
        "● LIVE CATALOG — CelesTrak"
    )

else:

    st.warning(
        "● CACHE MODE — CelesTrak unavailable"
    )


# ============================================================
# KPIs
# ============================================================

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Objects",
    summary["objects"]
)

c2.metric(
    "State rows",
    summary["state_rows"]
)

c3.metric(
    "Pairs",
    summary["possible_pairs"]
)

c4.metric(
    "Coarse candidates",
    summary["coarse_candidates"]
)

c5.metric(
    "True conjunctions",
    summary["true_conjunctions"]
)

c6.metric(
    "Audit",
    "PASS"
    if summary["audit_pass"]
    else "FAIL"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🛰️ ORBITAL OBJECTS",
        "⚠️ CONJUNCTIONS",
        "📊 RISK",
        "🔬 ANOMALIES",
        "🧪 AUDIT",
    ]
)


# ============================================================
# OBJECTS
# ============================================================

with tab1:

    st.subheader(
        "Orbital State Engine"
    )

    st.dataframe(
        r["state_summary"],
        use_container_width=True,
        hide_index=True,
    )

    states = r["states"]

    if not states.empty:

        st.subheader(
            "Altitude evolution"
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        for name, g in states.groupby(
            "name"
        ):

            g = g.sort_values(
                "grid_index"
            )

            altitude = (
                (
                    g[
                        [
                            "x_km",
                            "y_km",
                            "z_km",
                        ]
                    ]
                    ** 2
                ).sum(axis=1)
                ** 0.5
                - 6378.137
            )

            ax.plot(
                range(len(g)),
                altitude,
                label=name,
                alpha=0.55,
            )

        ax.set_xlabel(
            "Propagation step"
        )

        ax.set_ylabel(
            "Altitude [km]"
        )

        ax.grid(
            alpha=0.25
        )

        st.pyplot(
            fig,
            clear_figure=True
        )


# ============================================================
# CONJUNCTIONS
# ============================================================

with tab2:

    st.subheader(
        "Conjunction Detection"
    )

    refined = r["refined"]

    if refined.empty:

        st.success(
            "No conjunction candidates after refinement."
        )

    else:

        display = refined[
            [
                "object_a",
                "object_b",
                "miss_distance_km",
                "relative_velocity_km_s",
                "tca_utc",
                "tca_optimizer_success",
            ]
        ].copy()

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        true_events = r[
            "true_conjunctions"
        ]

        if not true_events.empty:

            st.error(
                f"{len(true_events)} "
                "event(s) below conjunction threshold."
            )

            st.dataframe(
                true_events,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# RISK
# ============================================================

with tab3:

    st.subheader(
        "Demonstration Risk Engine"
    )

    st.warning(
        """
        IMPORTANT: risk scores and Monte Carlo envelopes
        are demonstration outputs only.

        ASTRA-Q does NOT calculate operational collision
        probability Pc.
        """
    )

    risk = r["risk"]

    if risk.empty:

        st.info(
            "No true conjunction events."
        )

    else:

        st.dataframe(
            risk[
                [
                    "object_a",
                    "object_b",
                    "miss_distance_km",
                    "mc_p05_km",
                    "mc_median_km",
                    "mc_p95_km",
                    "risk_score_demo",
                    "risk_level",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ANOMALIES
# ============================================================

with tab4:

    st.subheader(
        "Orbital Anomaly Engine"
    )

    anomalies = r["anomalies"]

    if anomalies.empty:

        st.info(
            "No anomaly data."
        )

    else:

        anomaly_count = int(
            anomalies[
                "anomalous"
            ].sum()
        )

        if anomaly_count == 0:

            st.success(
                "No anomalous orbital samples detected."
            )

        else:

            st.warning(
                f"{anomaly_count} anomalous samples detected."
            )

        st.dataframe(
            anomalies.sort_values(
                "anomaly_index",
                ascending=False
            ).head(100),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# AUDIT
# ============================================================

with tab5:

    st.subheader(
        "SSA Structural Audit"
    )

    audit = r["audit"]

    audit_rows = []

    for key, value in audit.items():

        if isinstance(value, bool):

            status = (
                "PASS"
                if value
                else "FAIL"
            )

        else:

            status = str(value)

        audit_rows.append(
            {
                "Check": key,
                "Result": status,
            }
        )

    audit_df = pd.DataFrame(
        audit_rows
    )

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

    if r["audit_pass"]:

        st.success(
            "FINAL SSA AUDIT PASS"
        )

    else:

        st.error(
            "FINAL SSA AUDIT FAIL"
        )


# ============================================================
# DOWNLOADS
# ============================================================

st.divider()

st.subheader(
    "Generated ASTRA-Q Data"
)

generated_dir = "data/generated"

if os.path.exists(generated_dir):

    files = sorted(
        os.listdir(
            generated_dir
        )
    )

    cols = st.columns(
        min(4, max(1, len(files)))
    )

    for i, filename in enumerate(files):

        path = os.path.join(
            generated_dir,
            filename
        )

        with open(
            path,
            "rb"
        ) as f:

            data = f.read()

        cols[
            i % len(cols)
        ].download_button(
            filename,
            data=data,
            file_name=filename,
            key=f"download_{filename}",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"ASTRA-Q SSA v{VERSION} | "
    "SGP4 Orbital Intelligence Demonstrator"
)

st.caption(
    "Operational collision probability Pc is not calculated. "
    "Risk outputs are demonstration-only."
)
