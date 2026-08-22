import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from sgp4.api import Satrec, jday
from sgp4 import omm


# ============================================================
# ASTRA-Q SSA
# LOCAL OMM -> SGP4 -> CONJUNCTION SCREENING
# ============================================================

st.set_page_config(
    page_title="ASTRA-Q SSA",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CATALOG_FILE = DATA_DIR / "stations_omm.json"


# ============================================================
# PARAMETERS
# ============================================================

EARTH_RADIUS_KM = 6378.137

DEFAULT_HORIZON_H = 24
DEFAULT_STEP_MIN = 5

DEFAULT_SCREENING_KM = 50.0
DEFAULT_CONJUNCTION_KM = 25.0

COLOCATION_DISTANCE_KM = 5.0
COLOCATION_VREL_KM_S = 0.050


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 42px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.7;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">ASTRA-Q SSA</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Space Situational Awareness & Orbital Intelligence'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# OMM LOADER
# ============================================================

@st.cache_data
def load_omm_catalog():

    if not CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"FILE NOT FOUND:\n{CATALOG_FILE}"
        )

    with open(
        CATALOG_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # CelesTrak JSON normalmente = lista di dict
    # --------------------------------------------------------

    if isinstance(data, list):

        records = data

    elif isinstance(data, dict):

        if "data" in data:
            records = data["data"]

        elif "objects" in data:
            records = data["objects"]

        else:
            records = [data]

    else:

        raise ValueError(
            "stations_omm.json non contiene una lista/dizionario valido."
        )

    return records


# ============================================================
# OMM -> SATREC
# ============================================================

def build_satrec(record):

    """
    Converte un record OMM CelesTrak direttamente
    in un oggetto SGP4 Satrec.

    NON richiede TLE_LINE1/TLE_LINE2.
    """

    sat = Satrec()

    # sgp4.omm.initialize() gestisce direttamente
    # MEAN_MOTION, ECCENTRICITY, INCLINATION, ecc.
    omm.initialize(
        sat,
        record,
    )

    return sat


# ============================================================
# SAFE FIELD
# ============================================================

def field(record, *names, default=None):

    for name in names:

        if name in record:

            value = record[name]

            if value is not None:
                return value

    return default


# ============================================================
# LOAD
# ============================================================

try:

    records = load_omm_catalog()

except Exception as e:

    st.error("ERRORE CARICAMENTO CATALOGO")

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("ASTRA-Q Configuration")

max_objects = st.sidebar.slider(
    "Maximum objects",
    min_value=2,
    max_value=min(40, len(records)),
    value=min(22, len(records)),
)

horizon_h = st.sidebar.slider(
    "Propagation horizon [h]",
    1,
    72,
    DEFAULT_HORIZON_H,
)

step_min = st.sidebar.slider(
    "Propagation step [min]",
    1,
    30,
    DEFAULT_STEP_MIN,
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


# ============================================================
# CATALOG VALIDATION
# ============================================================

catalog = []

failures = []

for record in records:

    try:

        name = field(
            record,
            "OBJECT_NAME",
            "OBJECT",
            "NAME",
            default="UNKNOWN",
        )

        norad = field(
            record,
            "NORAD_CAT_ID",
            "NORAD_ID",
            "NORAD",
        )

        if norad is None:
            raise ValueError(
                "NORAD_CAT_ID mancante"
            )

        # TEST REALE: costruzione Satrec
        sat = build_satrec(record)

        catalog.append(
            {
                "name": str(name),
                "norad_id": int(norad),
                "record": record,
                "sat": sat,
            }
        )

    except Exception as e:

        failures.append(
            {
                "name": field(
                    record,
                    "OBJECT_NAME",
                    default="UNKNOWN",
                ),
                "error": str(e),
            }
        )


catalog = catalog[:max_objects]


# ============================================================
# STATUS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "OMM records",
        len(records),
    )

with c2:
    st.metric(
        "Valid SGP4",
        len(catalog),
    )

with c3:
    st.metric(
        "Failures",
        len(failures),
    )

with c4:
    st.metric(
        "Data source",
        "LOCAL OMM",
    )


if len(catalog) == 0:

    st.error(
        "NESSUN SATELLITE È STATO CONVERTITO IN SGP4."
    )

    if failures:

        st.dataframe(
            pd.DataFrame(failures),
            use_container_width=True,
        )

    st.stop()


st.success(
    f"LOCAL OMM CATALOG READY — {len(catalog)} objects"
)


# ============================================================
# CATALOG TABLE
# ============================================================

st.subheader("Orbital Catalog")

catalog_df = pd.DataFrame(
    [
        {
            "Object": x["name"],
            "NORAD ID": x["norad_id"],
        }
        for x in catalog
    ]
)

st.dataframe(
    catalog_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TIME GRID
# ============================================================

def make_time_grid():

    start = datetime.now(
        timezone.utc
    )

    n_steps = (
        int(
            horizon_h * 60 / step_min
        )
        + 1
    )

    return pd.date_range(
        start=start,
        periods=n_steps,
        freq=f"{step_min}min",
        tz="UTC",
    ).to_pydatetime()


# ============================================================
# PROPAGATION
# ============================================================

def propagate_satellite(
    sat,
    times,
):

    positions = []
    velocities = []
    errors = []

    for t in times:

        jd, fr = jday(
            t.year,
            t.month,
            t.day,
            t.hour,
            t.minute,
            t.second
            + t.microsecond / 1e6,
        )

        error, r, v = sat.sgp4(
            jd,
            fr,
        )

        errors.append(error)

        if error != 0:

            positions.append(
                [np.nan, np.nan, np.nan]
            )

            velocities.append(
                [np.nan, np.nan, np.nan]
            )

        else:

            positions.append(r)
            velocities.append(v)

    return (
        np.asarray(positions),
        np.asarray(velocities),
        np.asarray(errors),
    )


# ============================================================
# RUN
# ============================================================

st.subheader("SSA Processing")

run = st.button(
    "▶ RUN ASTRA-Q SSA",
    type="primary",
    use_container_width=True,
)


if run:

    times = make_time_grid()

    states = {}

    summary = []

    progress = st.progress(0)

    for i, obj in enumerate(catalog):

        sat = obj["sat"]

        pos, vel, errors = propagate_satellite(
            sat,
            times,
        )

        radius = np.linalg.norm(
            pos,
            axis=1,
        )

        altitude = (
            radius
            - EARTH_RADIUS_KM
        )

        speed = np.linalg.norm(
            vel,
            axis=1,
        )

        states[obj["norad_id"]] = {
            "name": obj["name"],
            "norad_id": obj["norad_id"],
            "times": times,
            "positions": pos,
            "velocities": vel,
            "errors": errors,
        }

        summary.append(
            {
                "object": obj["name"],
                "norad_id": obj["norad_id"],
                "altitude_min_km": np.nanmin(
                    altitude
                ),
                "altitude_max_km": np.nanmax(
                    altitude
                ),
                "altitude_mean_km": np.nanmean(
                    altitude
                ),
                "speed_mean_km_s": np.nanmean(
                    speed
                ),
                "state_rows": len(times),
                "sgp4_errors": int(
                    np.sum(errors != 0)
                ),
            }
        )

        progress.progress(
            (i + 1) / len(catalog)
        )

    progress.empty()

    summary_df = pd.DataFrame(summary)


    # ========================================================
    # PAIR ANALYSIS
    # ========================================================

    ids = list(states.keys())

    pairs = []

    for i in range(len(ids)):

        for j in range(
            i + 1,
            len(ids),
        ):

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

            distance = np.linalg.norm(
                dp,
                axis=1,
            )

            relative_velocity = np.linalg.norm(
                dv,
                axis=1,
            )

            finite = np.isfinite(
                distance
            )

            if not finite.any():
                continue

            finite_idx = np.where(
                finite
            )[0]

            local_idx = np.argmin(
                distance[finite_idx]
            )

            k = int(
                finite_idx[local_idx]
            )

            dmin = float(
                distance[k]
            )

            vrel = float(
                relative_velocity[k]
            )

            colocated = (
                dmin
                <= COLOCATION_DISTANCE_KM
                and
                vrel
                <= COLOCATION_VREL_KM_S
            )

            pairs.append(
                {
                    "object_a": a["name"],
                    "object_b": b["name"],
                    "norad_id_a": a["norad_id"],
                    "norad_id_b": b["norad_id"],
                    "min_grid_distance_km": dmin,
                    "relative_velocity_km_s": vrel,
                    "grid_index": k,
                    "grid_time_utc": str(
                        a["times"][k]
                    ),
                    "colocated": bool(
                        colocated
                    ),
                }
            )


    # ========================================================
    # CRITICAL FIX
    # ========================================================

    pair_columns = [
        "object_a",
        "object_b",
        "norad_id_a",
        "norad_id_b",
        "min_grid_distance_km",
        "relative_velocity_km_s",
        "grid_index",
        "grid_time_utc",
        "colocated",
    ]

    pairs_df = pd.DataFrame(
        pairs,
        columns=pair_columns,
    )


    # ========================================================
    # SAFE FILTER
    # ========================================================

    if pairs_df.empty:

        colocated_df = pd.DataFrame(
            columns=pair_columns
        )

        independent_df = pd.DataFrame(
            columns=pair_columns
        )

    else:

        colocated_df = pairs_df[
            pairs_df["colocated"].astype(bool)
        ].copy()

        independent_df = pairs_df[
            ~pairs_df["colocated"].astype(bool)
        ].copy()


    # ========================================================
    # SCREENING
    # ========================================================

    if independent_df.empty:

        candidates = independent_df.copy()

    else:

        candidates = independent_df[
            independent_df[
                "min_grid_distance_km"
            ]
            <= screening_km
        ].copy()

        candidates = candidates.sort_values(
            "min_grid_distance_km"
        )


    # ========================================================
    # CONJUNCTION
    # ========================================================

    if candidates.empty:

        conjunctions = candidates.copy()

    else:

        conjunctions = candidates[
            candidates[
                "min_grid_distance_km"
            ]
            <= conjunction_km
        ].copy()


    # ========================================================
    # SESSION
    # ========================================================

    st.session_state["states"] = states
    st.session_state["summary"] = summary_df
    st.session_state["pairs"] = pairs_df
    st.session_state["colocated"] = colocated_df
    st.session_state["independent"] = independent_df
    st.session_state["candidates"] = candidates
    st.session_state["conjunctions"] = conjunctions


# ============================================================
# RESULTS
# ============================================================

if "summary" in st.session_state:

    states = st.session_state["states"]
    summary_df = st.session_state["summary"]
    pairs_df = st.session_state["pairs"]
    colocated_df = st.session_state["colocated"]
    independent_df = st.session_state["independent"]
    candidates = st.session_state["candidates"]
    conjunctions = st.session_state["conjunctions"]


    # ========================================================
    # DASHBOARD
    # ========================================================

    st.divider()

    st.subheader("ASTRA-Q Mission Dashboard")

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Objects",
        len(states),
    )

    k2.metric(
        "Pairs",
        len(pairs_df),
    )

    k3.metric(
        "Co-located",
        len(colocated_df),
    )

    k4.metric(
        "Screened",
        len(candidates),
    )

    k5.metric(
        "Conjunctions",
        len(conjunctions),
    )


    # ========================================================
    # CONJUNCTION
    # ========================================================

    st.subheader(
        "Conjunction Screening"
    )

    if conjunctions.empty:

        st.success(
            "NO CONJUNCTION BELOW THRESHOLD"
        )

    else:

        st.warning(
            f"{len(conjunctions)} conjunction candidate(s)"
        )

        st.dataframe(
            conjunctions,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # ALL SCREENING CANDIDATES
    # ========================================================

    st.subheader(
        "Top Screening Candidates"
    )

    if candidates.empty:

        st.info(
            "No candidates inside screening distance."
        )

    else:

        st.dataframe(
            candidates.head(20),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # STATE SUMMARY
    # ========================================================

    st.subheader(
        "Orbital State Engine"
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # ALTITUDE
    # ========================================================

    st.subheader(
        "Altitude Evolution"
    )

    altitude_data = []

    for state in states.values():

        radius = np.linalg.norm(
            state["positions"],
            axis=1,
        )

        altitude = (
            radius
            - EARTH_RADIUS_KM
        )

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

        chart = altitude_df.pivot(
            index="time",
            columns="object",
            values="altitude_km",
        )

        st.line_chart(
            chart,
            height=450,
        )


    # ========================================================
    # PAIR AUDIT
    # ========================================================

    st.subheader(
        "Object Relationship Audit"
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Total pairs",
        len(pairs_df),
    )

    a2.metric(
        "Co-located",
        len(colocated_df),
    )

    a3.metric(
        "Independent",
        len(independent_df),
    )


    # ========================================================
    # STRUCTURAL AUDIT
    # ========================================================

    st.subheader(
        "ASTRA-Q Structural Audit"
    )

    expected_pairs = (
        len(states)
        * (len(states) - 1)
        // 2
    )

    checks = {

        "objects_nonzero":
            len(states) > 0,

        "states_nonzero":
            len(summary_df) > 0,

        "state_schema_valid":
            all(
                x in summary_df.columns
                for x in [
                    "object",
                    "norad_id",
                    "altitude_min_km",
                    "altitude_max_km",
                    "state_rows",
                ]
            ),

        "state_rows_expected":
            (
                len(times)
                if "times" in locals()
                else 0
            ),

        "state_rows_actual":
            int(
                summary_df["state_rows"].sum()
            )
            if not summary_df.empty
            else 0,

        "pair_count_consistency":
            len(pairs_df)
            == expected_pairs,

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

        "sgp4_errors_absent":
            int(
                summary_df["sgp4_errors"].sum()
            )
            == 0,

        "colocated_excluded":
            (
                len(
                    independent_df[
                        independent_df["colocated"]
                        == True
                    ]
                )
                == 0
            ),

        "one_grid_minimum_per_pair":
            len(pairs_df)
            == expected_pairs,
    }


    # Actual boolean audit
    audit_boolean = {

        "objects_nonzero":
            checks["objects_nonzero"],

        "states_nonzero":
            checks["states_nonzero"],

        "state_schema_valid":
            checks["state_schema_valid"],

        "pair_count_consistency":
            checks["pair_count_consistency"],

        "finite_positions":
            checks["finite_positions"],

        "finite_velocities":
            checks["finite_velocities"],

        "sgp4_errors_absent":
            checks["sgp4_errors_absent"],

        "colocated_excluded":
            checks["colocated_excluded"],

        "one_grid_minimum_per_pair":
            checks["one_grid_minimum_per_pair"],
    }


    audit_df = pd.DataFrame(
        [
            {
                "CHECK": key,
                "STATUS": (
                    "PASS"
                    if value
                    else "FAIL"
                ),
            }
            for key, value
            in audit_boolean.items()
        ]
    )

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )


    if all(
        audit_boolean.values()
    ):

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

    st.subheader(
        "Export Results"
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.download_button(
            "Download catalog.csv",
            catalog_df.to_csv(
                index=False
            ),
            "catalog.csv",
            "text/csv",
        )

    with d2:

        st.download_button(
            "Download state_summary.csv",
            summary_df.to_csv(
                index=False
            ),
            "state_summary.csv",
            "text/csv",
        )

    with d3:

        st.download_button(
            "Download conjunctions.csv",
            conjunctions.to_csv(
                index=False
            ),
            "conjunction_events.csv",
            "text/csv",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ASTRA-Q SSA — Demonstration / research platform. "
    "Local OMM catalog. SGP4 propagation. "
    "No operational collision probability (Pc) is calculated."
)
