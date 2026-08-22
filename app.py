import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except Exception:
    SGP4_AVAILABLE = False


# ============================================================
# ASTRA-Q SSA
# Streamlit demonstration
# ============================================================

st.set_page_config(
    page_title="ASTRA-Q SSA",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CATALOG_FILE = DATA_DIR / "stations_omm.json"

DEFAULT_HORIZON_H = 24
DEFAULT_STEP_MIN = 5
DEFAULT_SCREENING_KM = 50.0
DEFAULT_CONJUNCTION_KM = 25.0
DEFAULT_COLOCATION_KM = 5.0
DEFAULT_COLOCATION_VREL = 0.050


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .status-pass {
        color: #16a34a;
        font-weight: 700;
    }

    .status-warn {
        color: #d97706;
        font-weight: 700;
    }

    .status-fail {
        color: #dc2626;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">ASTRA-Q SSA</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Space Situational Awareness & Orbital Intelligence"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# CATALOG LOADER
# ============================================================

@st.cache_data
def load_catalog():

    if not CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"Catalogo non trovato: {CATALOG_FILE}"
        )

    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if "data" in raw:
            records = raw["data"]
        elif "objects" in raw:
            records = raw["objects"]
        else:
            records = [raw]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Formato stations_omm.json non valido.")

    valid = []

    for r in records:

        if not isinstance(r, dict):
            continue

        name = (
            r.get("OBJECT_NAME")
            or r.get("OBJECT")
            or r.get("name")
            or r.get("NAME")
        )

        norad = (
            r.get("NORAD_CAT_ID")
            or r.get("NORAD_ID")
            or r.get("norad_id")
            or r.get("NORAD")
        )

        line1 = r.get("TLE_LINE1")
        line2 = r.get("TLE_LINE2")

        # Alcuni cataloghi OMM contengono direttamente TLE
        if line1 and line2 and name and norad:

            valid.append(
                {
                    "name": str(name),
                    "norad_id": int(norad),
                    "line1": line1,
                    "line2": line2,
                    "raw": r,
                }
            )

            continue

        # Supporto OMM con elementi orbitali
        if name and norad:

            valid.append(
                {
                    "name": str(name),
                    "norad_id": int(norad),
                    "raw": r,
                }
            )

    return valid


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("ASTRA-Q Configuration")

horizon_h = st.sidebar.slider(
    "Propagation horizon [h]",
    min_value=1,
    max_value=72,
    value=DEFAULT_HORIZON_H,
)

step_min = st.sidebar.slider(
    "Propagation step [min]",
    min_value=1,
    max_value=30,
    value=DEFAULT_STEP_MIN,
)

screening_km = st.sidebar.number_input(
    "Screening distance [km]",
    min_value=1.0,
    max_value=500.0,
    value=DEFAULT_SCREENING_KM,
)

conjunction_km = st.sidebar.number_input(
    "Conjunction threshold [km]",
    min_value=1.0,
    max_value=100.0,
    value=DEFAULT_CONJUNCTION_KM,
)

max_objects = st.sidebar.slider(
    "Maximum objects",
    min_value=2,
    max_value=40,
    value=22,
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    catalog = load_catalog()

except Exception as e:

    st.error("Impossibile caricare il catalogo locale.")
    st.exception(e)

    st.stop()


catalog = catalog[:max_objects]


# ============================================================
# CATALOG STATUS
# ============================================================

st.success(
    f"LOCAL CATALOG MODE — {len(catalog)} objects loaded"
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Catalog objects", len(catalog))

with c2:
    st.metric("Source", "LOCAL JSON")

with c3:
    st.metric("Horizon", f"{horizon_h} h")

with c4:
    st.metric("Step", f"{step_min} min")


# ============================================================
# CATALOG TABLE
# ============================================================

st.subheader("Orbital Catalog")

catalog_table = pd.DataFrame(
    [
        {
            "Object": x["name"],
            "NORAD ID": x["norad_id"],
        }
        for x in catalog
    ]
)

st.dataframe(
    catalog_table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SGP4
# ============================================================

def create_satrec(item):

    if not SGP4_AVAILABLE:
        return None

    line1 = item.get("line1")
    line2 = item.get("line2")

    if not line1 or not line2:
        return None

    try:
        return Satrec.twoline2rv(line1, line2)
    except Exception:
        return None


# ============================================================
# PROPAGATION
# ============================================================

def propagate_satellite(sat, times):

    if sat is None:
        return None, None

    positions = []
    velocities = []

    for t in times:

        jd, fr = jday(
            t.year,
            t.month,
            t.day,
            t.hour,
            t.minute,
            t.second + t.microsecond / 1e6,
        )

        e, r, v = sat.sgp4(jd, fr)

        if e != 0:
            positions.append([np.nan] * 3)
            velocities.append([np.nan] * 3)
        else:
            positions.append(r)
            velocities.append(v)

    return (
        np.asarray(positions, dtype=float),
        np.asarray(velocities, dtype=float),
    )


# ============================================================
# PROPAGATION BUTTON
# ============================================================

st.subheader("Propagation Engine")

if not SGP4_AVAILABLE:

    st.warning(
        "sgp4 non disponibile. Verifica requirements.txt."
    )

else:

    if st.button(
        "▶ RUN ASTRA-Q SSA",
        type="primary",
        use_container_width=True,
    ):

        start = datetime.now(timezone.utc)

        times = pd.date_range(
            start=start,
            periods=int(horizon_h * 60 / step_min) + 1,
            freq=f"{step_min}min",
            tz="UTC",
        ).to_pydatetime()

        states = {}
        summaries = []

        progress = st.progress(0)

        for idx, item in enumerate(catalog):

            sat = create_satrec(item)

            if sat is None:
                continue

            pos, vel = propagate_satellite(
                sat,
                times,
            )

            if pos is None:
                continue

            states[item["norad_id"]] = {
                "name": item["name"],
                "norad_id": item["norad_id"],
                "times": times,
                "positions": pos,
                "velocities": vel,
            }

            radius = np.linalg.norm(pos, axis=1)

            altitude = radius - 6378.137

            speed = np.linalg.norm(vel, axis=1)

            summaries.append(
                {
                    "name": item["name"],
                    "norad_id": item["norad_id"],
                    "altitude_min_km": np.nanmin(altitude),
                    "altitude_max_km": np.nanmax(altitude),
                    "altitude_mean_km": np.nanmean(altitude),
                    "speed_mean_km_s": np.nanmean(speed),
                    "state_rows": len(times),
                }
            )

            progress.progress(
                (idx + 1) / len(catalog)
            )

        progress.empty()

        summary_df = pd.DataFrame(summaries)

        # ====================================================
        # PAIR SCREENING
        # ====================================================

        pairs = []

        ids = list(states.keys())

        for i in range(len(ids)):

            for j in range(i + 1, len(ids)):

                a = states[ids[i]]
                b = states[ids[j]]

                dp = (
                    a["positions"]
                    - b["positions"]
                )

                dv = (
                    a["velocities"]
                    - b["velocities"]
                )

                distances = np.linalg.norm(
                    dp,
                    axis=1,
                )

                velocities = np.linalg.norm(
                    dv,
                    axis=1,
                )

                k = int(np.nanargmin(distances))

                dmin = float(distances[k])

                vrel = float(velocities[k])

                colocated = (
                    dmin <= DEFAULT_COLOCATION_KM
                    and
                    vrel <= DEFAULT_COLOCATION_VREL
                )

                if colocated:
                    relation = "CO-LOCATED"
                else:
                    relation = "INDEPENDENT"

                pairs.append(
                    {
                        "object_a": a["name"],
                        "object_b": b["name"],
                        "norad_id_a": a["norad_id"],
                        "norad_id_b": b["norad_id"],
                        "min_distance_km": dmin,
                        "relative_velocity_km_s": vrel,
                        "colocated": bool(colocated),
                        "relation": relation,
                        "grid_index": k,
                        "time": a["times"][k],
                    }
                )

        pairs_df = pd.DataFrame(pairs)

        if pairs_df.empty:

            st.warning("Nessuna coppia disponibile.")

            independent_df = pd.DataFrame()

        else:

            independent_df = pairs_df[
                pairs_df["colocated"] == False
            ].copy()

        # ====================================================
        # CONJUNCTIONS
        # ====================================================

        if not independent_df.empty:

            candidates = independent_df[
                independent_df["min_distance_km"]
                <= screening_km
            ].copy()

            candidates = candidates.sort_values(
                "min_distance_km"
            )

        else:

            candidates = pd.DataFrame()

        true_conjunctions = candidates[
            candidates["min_distance_km"]
            <= conjunction_km
        ].copy()

        # ====================================================
        # SAVE RESULTS IN SESSION
        # ====================================================

        st.session_state["states"] = states
        st.session_state["summary"] = summary_df
        st.session_state["pairs"] = pairs_df
        st.session_state["candidates"] = candidates
        st.session_state["conjunctions"] = true_conjunctions

        st.session_state["run_time"] = start


# ============================================================
# RESULTS
# ============================================================

if "summary" not in st.session_state:

    st.info(
        "Premi **RUN ASTRA-Q SSA** per eseguire la propagazione."
    )

else:

    summary_df = st.session_state["summary"]
    pairs_df = st.session_state["pairs"]
    candidates = st.session_state["candidates"]
    conjunctions = st.session_state["conjunctions"]

    # ========================================================
    # KPIs
    # ========================================================

    st.subheader("SSA Mission Dashboard")

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric(
            "Objects",
            len(summary_df),
        )

    with k2:
        st.metric(
            "Pairs",
            len(pairs_df),
        )

    with k3:
        st.metric(
            "Screened",
            len(candidates),
        )

    with k4:
        st.metric(
            "Conjunctions",
            len(conjunctions),
        )

    with k5:

        if len(conjunctions) > 0:
            st.metric(
                "STATUS",
                "ALERT",
            )
        else:
            st.metric(
                "STATUS",
                "NOMINAL",
            )

    # ========================================================
    # CONJUNCTIONS
    # ========================================================

    st.subheader("Conjunction Analysis")

    if conjunctions.empty:

        st.success(
            "No conjunctions below the configured threshold."
        )

    else:

        st.warning(
            f"{len(conjunctions)} conjunction candidate(s) "
            "below threshold."
        )

        display = conjunctions[
            [
                "object_a",
                "object_b",
                "norad_id_a",
                "norad_id_b",
                "min_distance_km",
                "relative_velocity_km_s",
                "time",
            ]
        ].copy()

        display.columns = [
            "Object A",
            "Object B",
            "NORAD A",
            "NORAD B",
            "Miss distance [km]",
            "Relative velocity [km/s]",
            "TCA grid time",
        ]

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # SCREENING
    # ========================================================

    st.subheader("Screening Results")

    if candidates.empty:

        st.info("No screening candidates.")

    else:

        st.dataframe(
            candidates,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # ORBITAL SUMMARY
    # ========================================================

    st.subheader("Orbital State Summary")

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # ALTITUDE CHART
    # ========================================================

    st.subheader("Altitude Profile")

    states = st.session_state["states"]

    altitude_data = []

    for sid, state in states.items():

        radius = np.linalg.norm(
            state["positions"],
            axis=1,
        )

        altitude = radius - 6378.137

        for t, alt in zip(
            state["times"],
            altitude,
        ):

            altitude_data.append(
                {
                    "time": t,
                    "object": state["name"],
                    "altitude_km": alt,
                }
            )

    altitude_df = pd.DataFrame(
        altitude_data
    )

    if not altitude_df.empty:

        chart_df = altitude_df.pivot(
            index="time",
            columns="object",
            values="altitude_km",
        )

        st.line_chart(
            chart_df,
            height=450,
        )

    # ========================================================
    # RELATIONSHIP AUDIT
    # ========================================================

    st.subheader("Object Relationship Audit")

    if not pairs_df.empty:

        colocated_count = int(
            pairs_df["colocated"].sum()
        )

        independent_count = (
            len(pairs_df)
            - colocated_count
        )

        a1, a2, a3 = st.columns(3)

        with a1:
            st.metric(
                "Total pairs",
                len(pairs_df),
            )

        with a2:
            st.metric(
                "Co-located",
                colocated_count,
            )

        with a3:
            st.metric(
                "Independent",
                independent_count,
            )

    # ========================================================
    # AUDIT
    # ========================================================

    st.subheader("ASTRA-Q Structural Audit")

    checks = {
        "objects_nonzero":
            len(summary_df) > 0,

        "states_nonzero":
            len(states) > 0,

        "pair_count_consistency":
            len(pairs_df)
            == (
                len(summary_df)
                * (len(summary_df) - 1)
                // 2
            ),

        "finite_positions":
            all(
                np.isfinite(
                    s["positions"]
                ).all()
                for s in states.values()
            ),

        "finite_velocities":
            all(
                np.isfinite(
                    s["velocities"]
                ).all()
                for s in states.values()
            ),

        "state_rows_consistent":
            all(
                len(s["times"])
                == horizon_h * 60 // step_min + 1
                for s in states.values()
            ),

        "colocated_excluded":
            (
                pairs_df.empty
                or (
                    len(
                        candidates[
                            candidates["colocated"] == True
                        ]
                    )
                    == 0
                    if "colocated" in candidates.columns
                    else True
                )
            ),
    }

    audit_df = pd.DataFrame(
        [
            {
                "check": k,
                "PASS": bool(v),
            }
            for k, v in checks.items()
        ]
    )

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

    if all(checks.values()):

        st.success(
            "FINAL SSA AUDIT PASS"
        )

    else:

        st.error(
            "FINAL SSA AUDIT FAIL"
        )

    # ========================================================
    # DOWNLOADS
    # ========================================================

    st.subheader("Export")

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "Download state_summary.csv",
            summary_df.to_csv(
                index=False
            ),
            file_name="state_summary.csv",
            mime="text/csv",
        )

    with col2:

        st.download_button(
            "Download conjunctions.csv",
            conjunctions.to_csv(
                index=False
            ),
            file_name="conjunction_events.csv",
            mime="text/csv",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ASTRA-Q SSA — Demonstration platform. "
    "Propagation based on local catalog data. "
    "No operational collision probability is calculated."
)
