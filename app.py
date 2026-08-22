import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sgp4.api import Satrec, jday
except ImportError:
    st.error("Manca il pacchetto sgp4. Controlla requirements.txt.")
    st.stop()


# ============================================================
# ASTRA-Q SSA
# Streamlit application
# ============================================================

APP_VERSION = "ASTRA-Q SSA v8.6"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "stations_omm.json")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="ASTRA-Q SSA",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .risk-critical {
        color: #ff3333;
        font-weight: 700;
    }

    .risk-high {
        color: #ff9900;
        font-weight: 700;
    }

    .risk-low {
        color: #33aa66;
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
# LOAD LOCAL OMM
# ============================================================

@st.cache_data
def load_catalog():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Missing local catalog: {DATA_FILE}"
        )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "objects" in payload:
        records = payload["objects"]
        metadata = payload
    elif isinstance(payload, list):
        records = payload
        metadata = {}
    else:
        raise ValueError("Invalid stations_omm.json format")

    valid = []

    for r in records:

        if not isinstance(r, dict):
            continue

        if "NORAD_CAT_ID" not in r:
            continue

        valid.append(r)

    return valid, metadata


# ============================================================
# BUILD SGP4
# ============================================================

def build_satrec(record):

    # OMM JSON from CelesTrak
    sat = Satrec()

    sat.sgp4init(
        72,
        float(record["CLASSIFICATION"]),
        float(record["NORAD_CAT_ID"]),
        float(record["ELEMENT_SET_NO"]),
        float(record["EPOCH"]),
        float(record["MEAN_MOTION_DOT"]),
        float(record["MEAN_MOTION_DDOT"]),
        float(record["BSTAR"]),
        int(record["EPHEMERIS_TYPE"]),
        int(record["ELSET_NUM"]),
        float(record["INCLINATION"]),
        float(record["RA_OF_ASC_NODE"]),
        float(record["ECCENTRICITY"]),
        float(record["ARG_OF_PERICENTER"]),
        float(record["MEAN_ANOMALY"]),
        float(record["MEAN_MOTION"]),
        int(record["REV_AT_EPOCH"]),
    )

    return sat


# ============================================================
# SAFE OMM CONVERSION
# ============================================================

def make_satellite(record):

    try:
        sat = Satrec()

        # Prefer Satrec.twoline2rv if raw TLE exists
        if "TLE_LINE1" in record and "TLE_LINE2" in record:
            sat = Satrec.twoline2rv(
                record["TLE_LINE1"],
                record["TLE_LINE2"]
            )
            return sat

        # OMM fields
        sat.sgp4init(
            72,
            str(record.get("CLASSIFICATION", "U"))[0],
            int(record["NORAD_CAT_ID"]),
            float(record["EPOCH"]),
            float(record["MEAN_MOTION_DOT"]),
            float(record["MEAN_MOTION_DDOT"]),
            float(record["BSTAR"]),
            int(record.get("EPHEMERIS_TYPE", 0)),
            int(record.get("ELEMENT_SET_NO", 0)),
            np.deg2rad(float(record["INCLINATION"])),
            np.deg2rad(float(record["RA_OF_ASC_NODE"])),
            float(record["ECCENTRICITY"]),
            np.deg2rad(float(record["ARG_OF_PERICENTER"])),
            np.deg2rad(float(record["MEAN_ANOMALY"])),
            float(record["MEAN_MOTION"]) * 2.0 * np.pi / 1440.0,
            int(record["REV_AT_EPOCH"]),
        )

        return sat

    except Exception:
        return None


# ============================================================
# PROPAGATION
# ============================================================

def propagate_satellite(sat, start, hours=24, step_min=10):

    n = int(hours * 60 / step_min) + 1

    times = [
        start + timedelta(minutes=i * step_min)
        for i in range(n)
    ]

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

        error, r, v = sat.sgp4(jd, fr)

        if error != 0:
            positions.append([np.nan, np.nan, np.nan])
            velocities.append([np.nan, np.nan, np.nan])
        else:
            positions.append(r)
            velocities.append(v)

    return np.asarray(positions), np.asarray(velocities), times


# ============================================================
# LOAD
# ============================================================

try:

    records, metadata = load_catalog()

except Exception as e:

    st.error("Catalogo locale non disponibile.")

    st.code(str(e))

    st.info(
        "Inserisci data/stations_omm.json nel repository GitHub "
        "e riavvia l'app."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("ASTRA-Q CONTROL")

horizon = st.sidebar.slider(
    "Propagation horizon (hours)",
    1,
    48,
    24,
)

step = st.sidebar.selectbox(
    "Propagation step (minutes)",
    [5, 10, 15, 30],
    index=0,
)

threshold = st.sidebar.slider(
    "Conjunction threshold (km)",
    1.0,
    100.0,
    25.0,
)

max_objects = st.sidebar.slider(
    "Maximum objects",
    2,
    min(40, len(records)),
    min(22, len(records)),
)


# ============================================================
# CATALOG TABLE
# ============================================================

st.sidebar.markdown("---")

st.sidebar.metric(
    "LOCAL OMM OBJECTS",
    len(records)
)

st.sidebar.caption(
    "Catalog is loaded from GitHub/local repository."
)

if metadata.get("downloaded_utc"):
    st.sidebar.caption(
        "Catalog timestamp: "
        + str(metadata["downloaded_utc"])
    )


# ============================================================
# SELECT OBJECTS
# ============================================================

records = records[:max_objects]


# ============================================================
# PROPAGATE
# ============================================================

@st.cache_data(show_spinner=False)
def run_propagation(records_json, horizon, step):

    records_local = json.loads(records_json)

    output = []

    for idx, rec in enumerate(records_local):

        sat = make_satellite(rec)

        if sat is None:
            continue

        start = datetime.now(timezone.utc)

        pos, vel, times = propagate_satellite(
            sat,
            start,
            hours=horizon,
            step_min=step,
        )

        for k, t in enumerate(times):

            output.append({
                "object_index": idx,
                "name": rec.get(
                    "OBJECT_NAME",
                    rec.get("OBJECT_ID", f"OBJECT-{idx}")
                ),
                "norad_id": int(rec["NORAD_CAT_ID"]),
                "time": t,
                "x_km": pos[k, 0],
                "y_km": pos[k, 1],
                "z_km": pos[k, 2],
                "vx_km_s": vel[k, 0],
                "vy_km_s": vel[k, 1],
                "vz_km_s": vel[k, 2],
            })

    return pd.DataFrame(output)


records_json = json.dumps(
    records,
    sort_keys=True,
    default=str,
)


with st.spinner("Propagating orbital states..."):

    states = run_propagation(
        records_json,
        horizon,
        step,
    )


# ============================================================
# STATE VALIDATION
# ============================================================

if states.empty:

    st.error(
        "Nessun oggetto è stato propagato correttamente."
    )

    st.stop()


states["altitude_km"] = (
    np.sqrt(
        states["x_km"] ** 2
        + states["y_km"] ** 2
        + states["z_km"] ** 2
    )
    - 6378.137
)

states["speed_km_s"] = np.sqrt(
    states["vx_km_s"] ** 2
    + states["vy_km_s"] ** 2
    + states["vz_km_s"] ** 2
)


# ============================================================
# OBJECT SUMMARY
# ============================================================

summary = (
    states
    .groupby(["object_index", "name", "norad_id"])
    .agg(
        altitude_min_km=("altitude_km", "min"),
        altitude_max_km=("altitude_km", "max"),
        altitude_mean_km=("altitude_km", "mean"),
        speed_mean_km_s=("speed_km_s", "mean"),
        state_rows=("altitude_km", "count"),
    )
    .reset_index()
)


# ============================================================
# PAIR SCREENING
# ============================================================

def pair_screening(states_df, threshold_km):

    pairs = []

    grouped = {
        k: g.sort_values("time").reset_index(drop=True)
        for k, g in states_df.groupby("object_index")
    }

    object_ids = sorted(grouped.keys())

    for i in range(len(object_ids)):

        for j in range(i + 1, len(object_ids)):

            a = grouped[object_ids[i]]
            b = grouped[object_ids[j]]

            n = min(len(a), len(b))

            if n == 0:
                continue

            dr = (
                a[
                    ["x_km", "y_km", "z_km"]
                ].values[:n]
                -
                b[
                    ["x_km", "y_km", "z_km"]
                ].values[:n]
            )

            distance = np.linalg.norm(dr, axis=1)

            k = int(np.nanargmin(distance))

            dmin = float(distance[k])

            pairs.append({
                "object_a": a.iloc[0]["name"],
                "object_b": b.iloc[0]["name"],
                "object_index_a": int(object_ids[i]),
                "object_index_b": int(object_ids[j]),
                "min_distance_km": dmin,
                "time": a.iloc[k]["time"],
                "candidate": dmin <= threshold_km,
            })

    return pd.DataFrame(pairs)


pairs = pair_screening(
    states,
    threshold,
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

n_objects = len(summary)
n_states = len(states)
n_pairs = len(pairs)

candidates = (
    int(pairs["candidate"].sum())
    if not pairs.empty
    else 0
)

min_distance = (
    float(pairs["min_distance_km"].min())
    if not pairs.empty
    else np.nan
)


c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Objects",
    n_objects,
)

c2.metric(
    "State rows",
    n_states,
)

c3.metric(
    "Pairs",
    n_pairs,
)

c4.metric(
    "Candidates",
    candidates,
)

c5.metric(
    "Minimum distance",
    "—" if np.isnan(min_distance)
    else f"{min_distance:.2f} km",
)


# ============================================================
# STATUS
# ============================================================

if candidates == 0:

    st.success(
        "NO CONJUNCTION CANDIDATES BELOW THRESHOLD"
    )

else:

    st.warning(
        f"{candidates} conjunction candidate(s) detected"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🛰️ OBJECTS",
        "⚠️ CONJUNCTIONS",
        "📈 ORBITS",
        "🔍 AUDIT",
    ]
)


# ============================================================
# OBJECTS
# ============================================================

with tab1:

    st.subheader("Orbital State Engine")

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CONJUNCTIONS
# ============================================================

with tab2:

    st.subheader("Conjunction Screening")

    if pairs.empty:

        st.info("No pairs available.")

    else:

        candidates_df = pairs[
            pairs["candidate"]
        ].sort_values(
            "min_distance_km"
        )

        if candidates_df.empty:

            st.success(
                "No candidates below threshold."
            )

        else:

            st.dataframe(
                candidates_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                """
                **Risk model status**

                Operational collision probability (**Pc**) is
                **not calculated**.

                This dashboard performs geometric conjunction
                screening only.
                """
            )


# ============================================================
# ORBITAL VISUALIZATION
# ============================================================

with tab3:

    st.subheader("Altitude Evolution")

    chart_df = states[
        ["time", "name", "altitude_km"]
    ].pivot(
        index="time",
        columns="name",
        values="altitude_km",
    )

    st.line_chart(
        chart_df,
        height=500,
    )


# ============================================================
# AUDIT
# ============================================================

with tab4:

    st.subheader("ASTRA-Q Structural Audit")

    expected_rows = (
        n_objects
        * (int(horizon * 60 / step) + 1)
    )

    actual_rows = len(states)

    audit = {
        "version": APP_VERSION,
        "catalog_local": True,
        "catalog_objects": len(records),
        "objects_propagated": n_objects,
        "states_expected": expected_rows,
        "states_actual": actual_rows,
        "states_nonzero": actual_rows > 0,
        "state_schema_valid": all(
            c in states.columns
            for c in [
                "x_km",
                "y_km",
                "z_km",
                "vx_km_s",
                "vy_km_s",
                "vz_km_s",
            ]
        ),
        "finite_positions": bool(
            np.isfinite(
                states[
                    ["x_km", "y_km", "z_km"]
                ].values
            ).all()
        ),
        "finite_velocities": bool(
            np.isfinite(
                states[
                    ["vx_km_s", "vy_km_s", "vz_km_s"]
                ].values
            ).all()
        ),
        "pair_count": n_pairs,
        "conjunction_candidates": candidates,
        "operational_pc": False,
        "risk_model": "GEOMETRIC_SCREENING_ONLY",
    }

    audit["FINAL_SSA_AUDIT_PASS"] = all([
        audit["catalog_objects"] > 0,
        audit["objects_propagated"] > 0,
        audit["states_nonzero"],
        audit["state_schema_valid"],
        audit["states_actual"] == audit["states_expected"],
        audit["finite_positions"],
        audit["finite_velocities"],
    ])

    st.json(audit)

    if audit["FINAL_SSA_AUDIT_PASS"]:

        st.success(
            "FINAL SSA AUDIT PASS"
        )

    else:

        st.error(
            "FINAL SSA AUDIT FAIL"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    f"{APP_VERSION} | "
    "Local OMM catalog | "
    "SGP4 propagation | "
    "Geometric conjunction screening"
)
