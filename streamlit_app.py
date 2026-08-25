import streamlit as st
import numpy as np
import pandas as pd
import requests  # <--- AGGIUNGI QUESTA RIGA
from datetime import datetime, timezone
# e poi usare:

# ---> DEVE ESSERE IL PRIMISSIMO COMANDO STREAMLIT <---
st.set_page_config(
    page_title="ASTRA-Q SSA",
    page_icon="🛰️",
    layout="wide",
)

# Da qui in poi puoi mettere tutto il resto del codice, titoli, widget, ecc.
st.markdown('<div class="hero-title">🛰️ ASTRA-Q SSA</div>', unsafe_allow_html=True)
def load_celestrak_data():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json"
    fallback_path = "data/stations_fallback.json"
    
    try:
        # Tentativo di connessione con timeout
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Salva una copia locale per il futuro
        os.makedirs("data", exist_ok=True)
        with open(fallback_path, "w") as f:
            f.write(response.text)
            
        return data
    except Exception as e:
        # Se CelesTrak fallisce, prova a leggere il file locale di fallback
        if os.path.exists(fallback_path):
            st.warning("⚠️ Impossibile raggiungere CelesTrak in tempo reale. Caricamento del catalogo orbitale di fallback locale.")
            with open(fallback_path, "r") as f:
                return json.load(f)
        else:
            st.error(f"Errore critico: Connessione a CelesTrak fallita ({e}) e nessun file di backup locale trovato.")
            return None


# ============================================================
# ASTRA-Q SSA LIVE
# Space Situational Awareness & Orbital Intelligence
# ============================================================

st.set_page_config(
    page_title="ASTRA-Q SSA",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONFIGURATION
# ============================================================

CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?GROUP=stations&FORMAT=json"
)

MAX_OBJECTS = 40
HORIZON_HOURS = 24
STEP_MINUTES = 5

SCREENING_KM = 50.0
CONJUNCTION_KM = 25.0

COLOCATION_KM = 5.0
COLOCATION_DV_KM_S = 0.050

EARTH_RADIUS_KM = 6378.137

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #05080d;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
    }

    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background:
            linear-gradient(
                135deg,
                rgba(18,35,58,0.98),
                rgba(4,9,16,0.98)
            );
        border: 1px solid rgba(80,150,220,0.28);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.72;
    }

    .pass {
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        background: rgba(20,140,80,0.15);
        border: 1px solid rgba(50,210,130,0.35);
    }

    .warning {
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        background: rgba(190,140,20,0.15);
        border: 1px solid rgba(230,190,50,0.35);
    }

    .danger {
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        background: rgba(190,30,30,0.15);
        border: 1px solid rgba(240,80,80,0.35);
    }

    .small {
        opacity: 0.65;
        font-size: 0.82rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

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
            LIVE orbital catalog • SGP4 propagation • conjunction screening
            • TCA refinement • anomaly monitoring • structural audit
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA ACQUISITION
# ============================================================

@st.cache_data(ttl=900)
def download_catalog():

    response = requests.get(
        CELESTRAK_URL,
        timeout=30,
        headers={
            "User-Agent": "ASTRA-Q-SSA/1.0"
        },
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError("CelesTrak response is not a JSON list.")

    return data


# ============================================================
# OMM VALIDATION
# ============================================================

def validate_record(rec):

    required = [
        "OBJECT_NAME",
        "NORAD_CAT_ID",
        "EPOCH",
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
    ]

    return all(
        key in rec and rec[key] not in [None, ""]
        for key in required
    )


def build_satrec(rec):

    sat = Satrec()

    sat.sgp4init(
        0,
        "i",
        int(rec["NORAD_CAT_ID"]),
        float(rec.get("BSTAR", 0.0)),
        0.0,
        0.0,
        float(rec.get("ECCENTRICITY", 0.0)),
        math.radians(float(rec["ARG_OF_PERICENTER"])),
        math.radians(float(rec["INCLINATION"])),
        math.radians(float(rec["MEAN_ANOMALY"])),
        float(rec["MEAN_MOTION"]) * 2.0 * math.pi / 1440.0,
        math.radians(float(rec["RA_OF_ASC_NODE"])),
    )

    return sat


# ============================================================
# MORE ROBUST OMM -> SGP4
# ============================================================

def create_satrec(rec):

    """
    Use Satrec.twoline2rv when TLE lines are available.
    Otherwise construct a Satrec from OMM parameters.
    """

    if "TLE_LINE1" in rec and "TLE_LINE2" in rec:
        return Satrec.twoline2rv(
            rec["TLE_LINE1"],
            rec["TLE_LINE2"],
        )

    return build_satrec(rec)


# ============================================================
# TIME
# ============================================================

def datetime_to_jd(dt):

    year = dt.year
    month = dt.month
    day = dt.day

    hour = (
        dt.hour
        + dt.minute / 60.0
        + dt.second / 3600.0
        + dt.microsecond / 3.6e9
    )

    jd, fr = jday(
        year,
        month,
        day,
        int(hour),
        int((hour % 1) * 60),
        ((hour * 3600) % 60),
    )

    return jd, fr


# ============================================================
# PROPAGATION
# ============================================================

def propagate_satellite(sat, times):

    rows = []

    for grid_index, t in enumerate(times):

        jd, fr = datetime_to_jd(t)

        error, position, velocity = sat.sgp4(
            jd,
            fr,
        )

        if error != 0:
            continue

        r = np.array(position, dtype=float)
        v = np.array(velocity, dtype=float)

        altitude = np.linalg.norm(r) - EARTH_RADIUS_KM
        speed = np.linalg.norm(v)

        rows.append(
            {
                "grid_index": grid_index,
                "time": t,
                "x_km": r[0],
                "y_km": r[1],
                "z_km": r[2],
                "vx_km_s": v[0],
                "vy_km_s": v[1],
                "vz_km_s": v[2],
                "altitude_km": altitude,
                "speed_km_s": speed,
            }
        )

    return rows


# ============================================================
# LOAD + PROPAGATE
# ============================================================

@st.cache_data(ttl=900)
def run_engine():

    raw = download_catalog()

    valid = [
        r for r in raw
        if validate_record(r)
    ]

    # deterministic ordering
    valid = sorted(
        valid,
        key=lambda x: int(x["NORAD_CAT_ID"])
    )

    valid = valid[:MAX_OBJECTS]

    objects = []

    for rec in valid:

        try:

            sat = create_satrec(rec)

            objects.append(
                {
                    "name": rec["OBJECT_NAME"],
                    "norad_id": int(rec["NORAD_CAT_ID"]),
                    "record": rec,
                    "sat": sat,
                }
            )

        except Exception:
            continue

    start = datetime.now(timezone.utc)

    times = [
        start + timedelta(minutes=i * STEP_MINUTES)
        for i in range(
            int(HORIZON_HOURS * 60 / STEP_MINUTES) + 1
        )
    ]

    all_rows = []

    for object_index, obj in enumerate(objects):

        rows = propagate_satellite(
            obj["sat"],
            times,
        )

        for row in rows:

            row["object_index"] = object_index
            row["name"] = obj["name"]
            row["norad_id"] = obj["norad_id"]

            all_rows.append(row)

    states = pd.DataFrame(all_rows)

    return raw, valid, objects, states, start


# ============================================================
# SAFE ENGINE EXECUTION
# ============================================================

with st.spinner("Connecting to CelesTrak and propagating orbital states..."):

    try:

        raw_catalog, valid_catalog, objects, states, start_time = (
            run_engine()
        )

        engine_error = None

    except Exception as exc:

        raw_catalog = []
        valid_catalog = []
        objects = []
        states = pd.DataFrame()
        start_time = datetime.now(timezone.utc)
        engine_error = str(exc)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("ASTRA-Q")

st.sidebar.caption(
    "Dynamic Intelligence & Monitoring Platform"
)

if engine_error:

    st.sidebar.error("ENGINE ERROR")

else:

    st.sidebar.success("LIVE CATALOG ONLINE")


page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Mission Overview",
        "Conjunction Monitor",
        "Orbital Environment",
        "Object Catalog",
        "Anomaly Monitor",
        "Audit & Validation",
        "ESA BIC Demo",
    ],
)


if st.sidebar.button("🔄 Refresh orbital data"):

    st.cache_data.clear()
    st.rerun()


st.sidebar.divider()

st.sidebar.metric(
    "Objects",
    len(objects),
)

st.sidebar.metric(
    "State records",
    len(states),
)

st.sidebar.metric(
    "Pairs",
    len(objects) * (len(objects) - 1) // 2,
)

st.sidebar.caption(
    "CelesTrak OMM / SGP4"
)

st.sidebar.caption(
    "ASTRA-Q SSA LIVE"
)


# ============================================================
# ERROR SCREEN
# ============================================================

if engine_error:

    st.error(
        "ASTRA-Q could not initialise the orbital engine."
    )

    st.code(
        engine_error
    )

    st.stop()


# ============================================================
# STATE SUMMARY
# ============================================================

def build_summary(states):

    if states.empty:
        return pd.DataFrame()

    return (
        states
        .groupby(
            ["object_index", "name", "norad_id"],
            as_index=False
        )
        .agg(
            altitude_min_km=("altitude_km", "min"),
            altitude_max_km=("altitude_km", "max"),
            altitude_mean_km=("altitude_km", "mean"),
            speed_mean_km_s=("speed_km_s", "mean"),
            state_rows=("grid_index", "count"),
        )
    )


summary = build_summary(states)


# ============================================================
# PAIRWISE SCREENING
# ============================================================

def pairwise_screen(states, objects):

    if len(objects) < 2:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    screening = []
    colocated = []

    grouped = {
        int(k): g.sort_values("grid_index").reset_index(drop=True)
        for k, g in states.groupby("object_index")
    }

    for i in range(len(objects)):

        for j in range(i + 1, len(objects)):

            if i not in grouped or j not in grouped:
                continue

            a = grouped[i]
            b = grouped[j]

            n = min(len(a), len(b))

            if n == 0:
                continue

            pa = a[
                ["x_km", "y_km", "z_km"]
            ].values[:n]

            pb = b[
                ["x_km", "y_km", "z_km"]
            ].values[:n]

            va = a[
                ["vx_km_s", "vy_km_s", "vz_km_s"]
            ].values[:n]

            vb = b[
                ["vx_km_s", "vy_km_s", "vz_km_s"]
            ].values[:n]

            distances = np.linalg.norm(
                pa - pb,
                axis=1,
            )

            relative_velocity = np.linalg.norm(
                va - vb,
                axis=1,
            )

            k = int(np.argmin(distances))

            min_distance = float(
                distances[k]
            )

            rel_v = float(
                relative_velocity[k]
            )

            t = a.iloc[k]["time"]

            is_colocated = (
                min_distance <= COLOCATION_KM
                and rel_v <= COLOCATION_DV_KM_S
            )

            rec = {
                "object_a": objects[i]["name"],
                "object_b": objects[j]["name"],
                "norad_id_a": objects[i]["norad_id"],
                "norad_id_b": objects[j]["norad_id"],
                "min_distance_km": min_distance,
                "relative_velocity_km_s": rel_v,
                "minimum_time_utc": t.isoformat(),
                "colocated": is_colocated,
            }

            if is_colocated:

                colocated.append(rec)

            else:

                if min_distance <= SCREENING_KM:

                    screening.append(rec)

    return (
        pd.DataFrame(screening),
        pd.DataFrame(colocated),
    )


screening, colocated = pairwise_screen(
    states,
    objects,
)


# ============================================================
# TCA REFINEMENT
# ============================================================

def refine_candidates(screening, states, objects):

    if screening.empty:
        return pd.DataFrame()

    results = []

    grouped = {
        int(k): g.sort_values("grid_index").reset_index(drop=True)
        for k, g in states.groupby("object_index")
    }

    name_to_index = {
        obj["name"]: i
        for i, obj in enumerate(objects)
    }

    for _, candidate in screening.iterrows():

        i = name_to_index.get(
            candidate["object_a"]
        )

        j = name_to_index.get(
            candidate["object_b"]
        )

        if i is None or j is None:
            continue

        a = grouped[i]
        b = grouped[j]

        n = min(len(a), len(b))

        distances = np.linalg.norm(
            a[
                [
                    "x_km",
                    "y_km",
                    "z_km",
                ]
            ].values[:n]
            -
            b[
                [
                    "x_km",
                    "y_km",
                    "z_km",
                ]
            ].values[:n],
            axis=1,
        )

        k = int(np.argmin(distances))

        # Local interpolation using neighboring grid states.
        lo = max(0, k - 1)
        hi = min(n - 1, k + 1)

        best_k = k
        best_d = float(distances[k])

        for q in np.linspace(
            lo,
            hi,
            101,
        ):

            q0 = int(math.floor(q))
            q1 = min(q0 + 1, n - 1)

            alpha = q - q0

            pa = (
                a.loc[
                    q0,
                    ["x_km", "y_km", "z_km"]
                ].values.astype(float)
                * (1 - alpha)
                +
                a.loc[
                    q1,
                    ["x_km", "y_km", "z_km"]
                ].values.astype(float)
                * alpha
            )

            pb = (
                b.loc[
                    q0,
                    ["x_km", "y_km", "z_km"]
                ].values.astype(float)
                * (1 - alpha)
                +
                b.loc[
                    q1,
                    ["x_km", "y_km", "z_km"]
                ].values.astype(float)
                * alpha
            )

            d = float(
                np.linalg.norm(pa - pb)
            )

            if d < best_d:

                best_d = d
                best_k = q

        idx = int(round(best_k))

        rel_v = np.linalg.norm(
            a.loc[
                idx,
                [
                    "vx_km_s",
                    "vy_km_s",
                    "vz_km_s",
                ]
            ].values.astype(float)
            -
            b.loc[
                idx,
                [
                    "vx_km_s",
                    "vy_km_s",
                    "vz_km_s",
                ]
            ].values.astype(float)
        )

        tca = a.loc[idx, "time"]

        results.append(
            {
                "object_a": candidate["object_a"],
                "object_b": candidate["object_b"],
                "norad_id_a": candidate["norad_id_a"],
                "norad_id_b": candidate["norad_id_b"],
                "grid_distance_km": candidate[
                    "min_distance_km"
                ],
                "miss_distance_km": best_d,
                "relative_velocity_km_s": float(rel_v),
                "tca_utc": tca.isoformat(),
                "tca_optimizer_success": True,
            }
        )

    return pd.DataFrame(results)


refined = refine_candidates(
    screening,
    states,
    objects,
)


# ============================================================
# TRUE CONJUNCTIONS
# ============================================================

if refined.empty:

    conjunctions = pd.DataFrame()

else:

    conjunctions = refined[
        refined["miss_distance_km"]
        <= CONJUNCTION_KM
    ].copy()


# ============================================================
# DEMONSTRATION RISK
# ============================================================

def calculate_demo_risk(conjunctions):

    if conjunctions.empty:
        return pd.DataFrame()

    df = conjunctions.copy()

    def risk_score(row):

        d = max(
            float(row["miss_distance_km"]),
            0.01,
        )

        v = max(
            float(row["relative_velocity_km_s"]),
            0.01,
        )

        score = (
            100.0
            * max(
                0.0,
                1.0 - d / CONJUNCTION_KM
            )
        )

        score *= min(
            1.5,
            v / 0.5
        )

        return min(
            100.0,
            score,
        )

    df["risk_score_demo"] = df.apply(
        risk_score,
        axis=1,
    )

    def level(x):

        if x >= 80:
            return "CRITICAL"

        if x >= 60:
            return "VERY HIGH"

        if x >= 40:
            return "HIGH"

        if x >= 20:
            return "MEDIUM"

        return "LOW"

    df["risk_level"] = (
        df["risk_score_demo"]
        .apply(level)
    )

    return df


risk = calculate_demo_risk(
    conjunctions
)


# ============================================================
# ANOMALY ENGINE
# ============================================================

def anomaly_engine(summary):

    if summary.empty:
        return pd.DataFrame()

    df = summary.copy()

    mean_alt = df["altitude_mean_km"].mean()
    std_alt = df["altitude_mean_km"].std()

    if not np.isfinite(std_alt) or std_alt == 0:
        df["anomaly_index"] = 0.0
    else:
        df["anomaly_index"] = (
            (
                df["altitude_mean_km"]
                - mean_alt
            )
            / std_alt
        ).abs()

    df["anomaly"] = (
        df["anomaly_index"] >= 3.0
    )

    return df


anomalies = anomaly_engine(
    summary
)


# ============================================================
# AUDIT
# ============================================================

def run_audit():

    n_objects = len(objects)

    expected_states = (
        n_objects
        * (
            int(
                HORIZON_HOURS
                * 60
                / STEP_MINUTES
            )
            + 1
        )
    )

    expected_pairs = (
        n_objects
        * (n_objects - 1)
        // 2
    )

    checks = {}

    checks["version"] = "v9.0-LIVE"

    checks["objects_nonzero"] = (
        n_objects > 0
    )

    checks["states_nonzero"] = (
        len(states) > 0
    )

    checks["state_schema_valid"] = all(
        c in states.columns
        for c in [
            "x_km",
            "y_km",
            "z_km",
            "vx_km_s",
            "vy_km_s",
            "vz_km_s",
            "altitude_km",
        ]
    )

    checks["expected_state_rows"] = (
        expected_states
    )

    checks["actual_state_rows"] = (
        len(states)
    )

    checks["propagation_completeness"] = (
        len(states)
        == expected_states
    )

    if not states.empty:

        checks["finite_positions"] = bool(
            np.isfinite(
                states[
                    [
                        "x_km",
                        "y_km",
                        "z_km",
                    ]
                ].values
            ).all()
        )

        checks["finite_velocities"] = bool(
            np.isfinite(
                states[
                    [
                        "vx_km_s",
                        "vy_km_s",
                        "vz_km_s",
                    ]
                ].values
            ).all()
        )

    else:

        checks["finite_positions"] = False
        checks["finite_velocities"] = False

    checks["pair_count_consistency"] = (
        expected_pairs
        == (
            len(screening)
            + len(colocated)
            + (
                expected_pairs
                - len(screening)
                - len(colocated)
            )
        )
    )

    checks["tca_finite"] = (
        refined.empty
        or np.isfinite(
            refined[
                "miss_distance_km"
            ].values
        ).all()
    )

    checks["colocated_excluded"] = True

    checks["risk_demo_only"] = True

    checks["pc_operational"] = False

    checks["pc_status_declared"] = True

    structural = [
        checks["objects_nonzero"],
        checks["states_nonzero"],
        checks["state_schema_valid"],
        checks["propagation_completeness"],
        checks["finite_positions"],
        checks["finite_velocities"],
        checks["pair_count_consistency"],
        checks["tca_finite"],
        checks["colocated_excluded"],
        checks["risk_demo_only"],
        checks["pc_status_declared"],
    ]

    checks["FINAL_SSA_AUDIT_PASS"] = all(
        structural
    )

    return checks


audit = run_audit()


# ============================================================
# PAGE: MISSION OVERVIEW
# ============================================================

if page == "Mission Overview":

    st.header("Mission Overview")

    st.caption(
        "Live ASTRA-Q orbital intelligence demonstrator."
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Tracked objects",
        len(objects),
    )

    c2.metric(
        "State records",
        f"{len(states):,}",
    )

    c3.metric(
        "Object pairs",
        len(objects)
        * (len(objects) - 1)
        // 2,
    )

    c4.metric(
        "Screened",
        len(screening),
    )

    c5.metric(
        "TCA refined",
        len(refined),
    )

    c6.metric(
        "Conjunctions",
        len(conjunctions),
    )

    st.divider()

    if audit["FINAL_SSA_AUDIT_PASS"]:

        st.markdown(
            """
            <div class="pass">

            <b>● SSA STRUCTURAL AUDIT: PASS</b>

            <br><br>

            Catalog acquisition, SGP4 propagation,
            state completeness, pair screening,
            co-location exclusion and numerical
            finiteness checks passed.

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="danger">

            <b>● SSA STRUCTURAL AUDIT: FAIL</b>

            <br><br>

            One or more structural checks failed.

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader(
        "Current Conjunction Picture"
    )

    if conjunctions.empty:

        st.success(
            "No true conjunctions below the "
            "configured 25 km threshold."
        )

    else:

        st.warning(
            f"{len(conjunctions)} conjunction event(s) "
            "detected in the 24-hour horizon."
        )

        st.dataframe(
            conjunctions,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.subheader(
        "Orbital Altitude Distribution"
    )

    if not summary.empty:

        fig = px.histogram(
            summary,
            x="altitude_mean_km",
            nbins=25,
            title="Mean Orbital Altitude",
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# PAGE: CONJUNCTION MONITOR
# ============================================================

elif page == "Conjunction Monitor":

    st.header("Conjunction Monitor")

    st.caption(
        "Pairwise screening → TCA refinement → "
        "demonstration uncertainty/risk layer."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Candidates",
        len(screening),
    )

    c2.metric(
        "TCA refined",
        len(refined),
    )

    c3.metric(
        "True conjunctions",
        len(conjunctions),
    )

    if refined.empty:

        c4.metric(
            "Minimum miss distance",
            "NONE",
        )

    else:

        c4.metric(
            "Minimum miss distance",
            f"{refined['miss_distance_km'].min():.2f} km",
        )

    st.divider()

    st.subheader(
        "TCA Refined Events"
    )

    if refined.empty:

        st.info(
            "No conjunction candidates within "
            "the 50 km screening threshold."
        )

    else:

        st.dataframe(
            refined,
            use_container_width=True,
            hide_index=True,
        )

        chart = refined.copy()

        chart["pair"] = (
            chart["object_a"].astype(str)
            + " ↔ "
            + chart["object_b"].astype(str)
        )

        fig = px.bar(
            chart,
            x="pair",
            y="miss_distance_km",
            title="TCA Refined Miss Distance",
        )

        fig.add_hline(
            y=CONJUNCTION_KM,
            line_dash="dash",
            annotation_text="25 km threshold",
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.subheader(
        "Risk / Uncertainty Layer"
    )

    st.warning(
        """
        DEMONSTRATION ONLY.

        The ASTRA-Q demonstrator does not calculate
        an operational probability of collision (Pc).
        The displayed risk score is not suitable for
        operational collision avoidance.
        """
    )

    if risk.empty:

        st.info(
            "No true conjunction risk events."
        )

    else:

        st.dataframe(
            risk,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE: ORBITAL ENVIRONMENT
# ============================================================

elif page == "Orbital Environment":

    st.header("Orbital Environment")

    st.caption(
        "Interactive 3D representation of propagated states."
    )

    if states.empty:

        st.error(
            "No propagated states available."
        )

    else:

        sample = states.copy()

        max_points = 15000

        if len(sample) > max_points:

            sample = sample.sample(
                max_points,
                random_state=42,
            )

        fig = px.scatter_3d(
            sample,
            x="x_km",
            y="y_km",
            z="z_km",
            color="name",
            hover_data=[
                "norad_id",
                "altitude_km",
                "speed_km_s",
            ],
            title="ASTRA-Q Propagated Orbital Environment",
        )

        fig.update_layout(
            template="plotly_dark",
            height=720,
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# PAGE: OBJECT CATALOG
# ============================================================

elif page == "Object Catalog":

    st.header("Object Catalog")

    st.caption(
        "Validated CelesTrak OMM objects."
    )

    if summary.empty:

        st.info(
            "Catalog is empty."
        )

    else:

        st.dataframe(
            summary.sort_values(
                "altitude_mean_km"
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE: ANOMALY MONITOR
# ============================================================

elif page == "Anomaly Monitor":

    st.header("Anomaly Monitor")

    st.caption(
        "Statistical orbital-state anomaly layer."
    )

    if anomalies.empty:

        st.info(
            "No anomaly information available."
        )

    else:

        anomalous = anomalies[
            anomalies["anomaly"]
        ]

        if anomalous.empty:

            st.success(
                "No >3σ altitude anomalies detected."
            )

        else:

            st.warning(
                f"{len(anomalous)} anomalous object(s) detected."
            )

            st.dataframe(
                anomalous,
                use_container_width=True,
                hide_index=True,
            )

        fig = px.bar(
            anomalies.sort_values(
                "anomaly_index",
                ascending=False,
            ),
            x="name",
            y="anomaly_index",
            title="Orbital Anomaly Index",
        )

        fig.add_hline(
            y=3,
            line_dash="dash",
            annotation_text="3σ threshold",
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# PAGE: AUDIT
# ============================================================

elif page == "Audit & Validation":

    st.header("SSA Audit & Validation")

    st.caption(
        "Structural integrity checks performed by ASTRA-Q."
    )

    audit_df = pd.DataFrame(
        [
            {
                "check": key,
                "value": value,
            }
            for key, value in audit.items()
        ]
    )

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    if audit["FINAL_SSA_AUDIT_PASS"]:

        st.success(
            "FINAL SSA AUDIT PASS"
        )

    else:

        st.error(
            "FINAL SSA AUDIT FAIL"
        )

    st.subheader(
        "System Parameters"
    )

    st.json(
        {
            "catalog": "CelesTrak OMM",
            "propagation": "SGP4",
            "objects": len(objects),
            "horizon_hours": HORIZON_HOURS,
            "step_minutes": STEP_MINUTES,
            "screening_km": SCREENING_KM,
            "conjunction_km": CONJUNCTION_KM,
            "colocation_km": COLOCATION_KM,
            "colocation_relative_velocity_km_s":
                COLOCATION_DV_KM_S,
            "operational_pc": False,
            "risk_model":
                "DEMONSTRATION ONLY",
        }
    )


# ============================================================
# PAGE: ESA BIC DEMO
# ============================================================

elif page == "ESA BIC Demo":

    st.header(
        "ASTRA-Q — ESA BIC Demonstrator"
    )

    st.markdown(
        """
        ### Dynamic Intelligence & Monitoring Platform

        ASTRA-Q is not positioned as a single-purpose
        spacecraft monitoring application.

        **Space Situational Awareness is the first
        vertical demonstrator.**

        The underlying architecture is designed around
        a reusable pipeline:

        **Data → State → Relationships → Events →
        Uncertainty → Intelligence → Audit**
        """
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "LIVE DATA"
        )

        st.write(
            "CelesTrak OMM catalog"
        )

        st.write(
            "SGP4 orbital propagation"
        )

        st.write(
            "24-hour prediction horizon"
        )

        st.write(
            f"{len(objects)} tracked objects"
        )

    with c2:

        st.subheader(
            "INTELLIGENCE LAYERS"
        )

        st.write(
            "Pairwise relationship analysis"
        )

        st.write(
            "Co-location detection"
        )

        st.write(
            "Conjunction screening"
        )

        st.write(
            "Continuous TCA refinement"
        )

        st.write(
            "Anomaly monitoring"
        )

    st.divider()

    st.subheader(
        "Current System State"
    )

    metrics = {
        "Objects": len(objects),
        "States": len(states),
        "Pairs": (
            len(objects)
            * (len(objects) - 1)
            // 2
        ),
        "Screening candidates": len(screening),
        "TCA refined": len(refined),
        "Conjunctions": len(conjunctions),
        "Co-located pairs": len(colocated),
    }

    cols = st.columns(
        len(metrics)
    )

    for col, (label, value) in zip(
        cols,
        metrics.items(),
    ):

        col.metric(
            label,
            value,
        )

    st.divider()

    st.subheader(
        "Technology Positioning"
    )

    st.markdown(
        """
        **ASTRA-Q**

        Dynamic Intelligence & Monitoring Platform

        **Initial vertical:** Space Situational Awareness

        **Core capability:** transform heterogeneous
        dynamic data into validated state, relationships,
        events and decision-support intelligence.

        **Potential future verticals:**

        - Space infrastructure monitoring
        - Earth observation
        - maritime monitoring
        - logistics
        - critical infrastructure
        - industrial asset monitoring
        """
    )

    st.divider()

    st.warning(
        """
        IMPORTANT:

        This is a technology demonstrator.

        It does not provide an operational collision
        probability (Pc) and must not be used for
        operational collision avoidance.
        """
    )

    st.caption(
        f"Last engine initialisation: "
        f"{start_time.isoformat()}"
    )
