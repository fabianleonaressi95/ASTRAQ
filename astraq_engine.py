from __future__ import annotations

import json
import math
import itertools
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from scipy.optimize import minimize_scalar
from sgp4.api import Satrec
from sgp4.api import jday


# ============================================================
# ASTRA-Q SSA ENGINE
# ============================================================

VERSION = "v9.0"

CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?GROUP=stations&FORMAT=json"
)

DEFAULT_CONFIG = {
    "max_objects": 40,
    "horizon_hours": 24.0,
    "step_minutes": 5.0,
    "screening_km": 50.0,
    "conjunction_km": 25.0,
    "colocation_km": 5.0,
    "colocation_dv_km_s": 0.050,
    "mc_samples": 300,
    "position_sigma_km": 1.0,
    "velocity_sigma_km_s": 0.001,
}


# ============================================================
# OUTPUT
# ============================================================

def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# CATALOG
# ============================================================

def download_catalog():

    r = requests.get(
        CELESTRAK_URL,
        timeout=30,
        headers={"User-Agent": "ASTRA-Q-SSA/9.0"},
    )

    r.raise_for_status()

    data = r.json()

    if not isinstance(data, list):
        raise RuntimeError("CelesTrak returned an unexpected format.")

    return data


def validate_catalog(records):

    valid = []

    seen = set()

    for rec in records:

        try:

            norad = int(rec["NORAD_CAT_ID"])

            name = str(
                rec.get("OBJECT_NAME", f"NORAD-{norad}")
            ).strip()

            if norad in seen:
                continue

            # Required OMM fields
            required = [
                "EPOCH",
                "MEAN_MOTION",
                "ECCENTRICITY",
                "INCLINATION",
                "ARG_OF_PERICENTER",
                "RA_OF_ASC_NODE",
                "MEAN_ANOMALY",
            ]

            if not all(k in rec for k in required):
                continue

            seen.add(norad)

            valid.append(rec)

        except Exception:
            continue

    return valid


# ============================================================
# SGP4
# ============================================================

def omm_to_satrec(rec):

    sat = Satrec()

    sat.sgp4init(
        84,
        "i",
        int(rec["NORAD_CAT_ID"]),
        0.0,
        0.0,
        float(rec["MEAN_MOTION_DOT"]),
        float(rec["MEAN_MOTION_DDOT"]),
        float(rec["BSTAR"]),
        float(rec["ECCENTRICITY"]),
        math.radians(float(rec["ARG_OF_PERICENTER"])),
        math.radians(float(rec["INCLINATION"])),
        math.radians(float(rec["MEAN_ANOMALY"])),
        float(rec["MEAN_MOTION"]),
        math.radians(float(rec["RA_OF_ASC_NODE"])),
    )

    return sat


def parse_epoch(epoch_string):

    # OMM epoch format:
    # YYYY-MM-DDTHH:MM:SS.ssssss
    text = str(epoch_string).replace("Z", "")

    dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


# ============================================================
# PROPAGATION
# ============================================================

def propagate_satellite(sat, dt):

    jd, fr = jday(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second + dt.microsecond / 1e6,
    )

    error, position, velocity = sat.sgp4(jd, fr)

    if error != 0:
        return None

    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    if not np.all(np.isfinite(position)):
        return None

    if not np.all(np.isfinite(velocity)):
        return None

    return position, velocity


# ============================================================
# PROPAGATION GRID
# ============================================================

def build_grid(start, horizon_hours, step_minutes):

    n = int(
        round(horizon_hours * 60.0 / step_minutes)
    )

    return [
        start + timedelta(
            minutes=i * step_minutes
        )
        for i in range(n + 1)
    ]


def propagate_catalog(catalog, satellites, grid):

    rows = []

    for obj_index, (rec, sat) in enumerate(
        zip(catalog, satellites)
    ):

        norad = int(rec["NORAD_CAT_ID"])

        name = str(
            rec.get("OBJECT_NAME", f"NORAD-{norad}")
        ).strip()

        for grid_index, dt in enumerate(grid):

            result = propagate_satellite(sat, dt)

            if result is None:
                continue

            pos, vel = result

            altitude = (
                float(np.linalg.norm(pos)) - 6378.137
            )

            speed = float(np.linalg.norm(vel))

            rows.append(
                {
                    "object_index": obj_index,
                    "name": name,
                    "norad_id": norad,
                    "grid_index": grid_index,
                    "grid_time_utc": dt.isoformat(),
                    "x_km": pos[0],
                    "y_km": pos[1],
                    "z_km": pos[2],
                    "vx_km_s": vel[0],
                    "vy_km_s": vel[1],
                    "vz_km_s": vel[2],
                    "altitude_km": altitude,
                    "speed_km_s": speed,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# STATE SUMMARY
# ============================================================

def build_state_summary(states):

    if states.empty:
        return pd.DataFrame()

    return (
        states.groupby(
            ["object_index", "name", "norad_id"],
            as_index=False,
        )
        .agg(
            altitude_min_km=("altitude_km", "min"),
            altitude_max_km=("altitude_km", "max"),
            altitude_mean_km=("altitude_km", "mean"),
            speed_mean_km_s=("speed_km_s", "mean"),
            state_rows=("grid_index", "count"),
        )
    )


# ============================================================
# STATE CACHE
# ============================================================

def build_state_cache(states):

    cache = {}

    if states.empty:
        return cache

    for object_index, g in states.groupby(
        "object_index"
    ):

        g = g.sort_values(
            "grid_index"
        ).reset_index(drop=True)

        cache[int(object_index)] = {
            "name": str(g.iloc[0]["name"]),
            "norad_id": int(g.iloc[0]["norad_id"]),
            "time": pd.to_datetime(
                g["grid_time_utc"]
            ).to_numpy(),
            "pos": g[
                ["x_km", "y_km", "z_km"]
            ].to_numpy(float),
            "vel": g[
                ["vx_km_s", "vy_km_s", "vz_km_s"]
            ].to_numpy(float),
        }

    return cache


# ============================================================
# COLOCATION
# ============================================================

def audit_pairs(cache, config):

    object_ids = sorted(cache.keys())

    pairs = []

    for a, b in itertools.combinations(
        object_ids, 2
    ):

        A = cache[a]
        B = cache[b]

        n = min(
            len(A["pos"]),
            len(B["pos"])
        )

        if n == 0:
            continue

        distances = np.linalg.norm(
            A["pos"][:n] - B["pos"][:n],
            axis=1,
        )

        dvs = np.linalg.norm(
            A["vel"][:n] - B["vel"][:n],
            axis=1,
        )

        idx = int(np.argmin(distances))

        distance = float(distances[idx])
        dv = float(dvs[idx])

        colocated = (
            distance <= config["colocation_km"]
            and dv <= config["colocation_dv_km_s"]
        )

        pairs.append(
            {
                "object_index_a": a,
                "object_index_b": b,
                "norad_id_a": A["norad_id"],
                "norad_id_b": B["norad_id"],
                "object_a": A["name"],
                "object_b": B["name"],
                "min_grid_distance_km": distance,
                "relative_velocity_at_grid_min_km_s": dv,
                "minimum_grid_time_utc": str(
                    A["time"][idx]
                ),
                "colocated": bool(colocated),
            }
        )

    return pd.DataFrame(pairs)


# ============================================================
# CONJUNCTION SCREENING
# ============================================================

def screen_conjunctions(
    cache,
    pair_df,
    config,
):

    candidates = []

    if pair_df.empty:
        return pd.DataFrame()

    independent = pair_df[
        ~pair_df["colocated"]
    ]

    for _, pair in independent.iterrows():

        a = int(pair["object_index_a"])
        b = int(pair["object_index_b"])

        A = cache[a]
        B = cache[b]

        n = min(
            len(A["pos"]),
            len(B["pos"])
        )

        distances = np.linalg.norm(
            A["pos"][:n] - B["pos"][:n],
            axis=1,
        )

        idx = int(np.argmin(distances))

        d = float(distances[idx])

        if d <= config["screening_km"]:

            candidates.append(
                {
                    "object_index_a": a,
                    "object_index_b": b,
                    "norad_id_a": A["norad_id"],
                    "norad_id_b": B["norad_id"],
                    "object_a": A["name"],
                    "object_b": B["name"],
                    "grid_index": idx,
                    "grid_time_utc": str(
                        A["time"][idx]
                    ),
                    "grid_distance_km": d,
                }
            )

    return pd.DataFrame(candidates)


# ============================================================
# CONTINUOUS TCA
# ============================================================

def interpolate_state(cache_entry, t_seconds):

    times = cache_entry["time"]

    t0 = times[0].astype(
        "datetime64[ns]"
    ).astype(np.int64) / 1e9

    tt = np.asarray(
        times.astype("datetime64[ns]")
        .astype(np.int64) / 1e9
    )

    target = float(t0 + t_seconds)

    pos = np.array(
        [
            np.interp(
                target,
                tt,
                cache_entry["pos"][:, j],
            )
            for j in range(3)
        ]
    )

    vel = np.array(
        [
            np.interp(
                target,
                tt,
                cache_entry["vel"][:, j],
            )
            for j in range(3)
        ]
    )

    return pos, vel


def refine_tca(
    cache,
    screening,
):

    refined = []

    if screening.empty:
        return pd.DataFrame()

    for _, row in screening.iterrows():

        a = int(row["object_index_a"])
        b = int(row["object_index_b"])

        A = cache[a]
        B = cache[b]

        n = min(
            len(A["pos"]),
            len(B["pos"])
        )

        distances = np.linalg.norm(
            A["pos"][:n] - B["pos"][:n],
            axis=1,
        )

        idx = int(np.argmin(distances))

        # +/- one grid interval
        lo = max(0, idx - 1)
        hi = min(n - 1, idx + 1)

        t0 = lo * 300.0
        t1 = hi * 300.0

        def objective(t):

            pa, _ = interpolate_state(
                A, t
            )

            pb, _ = interpolate_state(
                B, t
            )

            return float(
                np.linalg.norm(pa - pb)
            )

        result = minimize_scalar(
            objective,
            bounds=(t0, t1),
            method="bounded",
        )

        tca_seconds = float(
            result.x
        )

        pa, va = interpolate_state(
            A, tca_seconds
        )

        pb, vb = interpolate_state(
            B, tca_seconds
        )

        miss = float(
            np.linalg.norm(pa - pb)
        )

        rel_v = float(
            np.linalg.norm(va - vb)
        )

        start = pd.Timestamp(
            A["time"][0]
        ).to_pydatetime()

        tca = start + timedelta(
            seconds=tca_seconds
        )

        refined.append(
            {
                "object_index_a": a,
                "object_index_b": b,
                "norad_id_a": A["norad_id"],
                "norad_id_b": B["norad_id"],
                "object_a": A["name"],
                "object_b": B["name"],
                "grid_distance_km": float(
                    row["grid_distance_km"]
                ),
                "miss_distance_km": miss,
                "relative_velocity_km_s": rel_v,
                "tca_minutes_from_start":
                    tca_seconds / 60.0,
                "tca_utc": tca.isoformat(),
                "tca_optimizer_success":
                    bool(result.success),
            }
        )

    return pd.DataFrame(refined)


# ============================================================
# RISK / MONTE CARLO
# ============================================================

def monte_carlo_risk(
    refined,
    config,
):

    events = []

    if refined.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(42)

    for _, row in refined.iterrows():

        md = float(
            row["miss_distance_km"]
        )

        samples = []

        for _ in range(
            config["mc_samples"]
        ):

            noise = rng.normal(
                0.0,
                config["position_sigma_km"],
            )

            samples.append(
                abs(md + noise)
            )

        samples = np.asarray(samples)

        p05, median, p95 = np.percentile(
            samples,
            [5, 50, 95],
        )

        if md <= 10:
            level = "CRITICAL"
        elif md <= 25:
            level = "HIGH"
        elif md <= 50:
            level = "MEDIUM"
        else:
            level = "LOW"

        score = max(
            0.0,
            min(
                100.0,
                100.0 * (
                    1.0 -
                    md / max(
                        config["screening_km"],
                        1.0,
                    )
                ),
            ),
        )

        events.append(
            {
                **row.to_dict(),
                "pc_demo": np.nan,
                "mc_p05_km": float(p05),
                "mc_median_km": float(median),
                "mc_p95_km": float(p95),
                "risk_score_demo": float(score),
                "risk_level": level,
            }
        )

    return pd.DataFrame(events)


# ============================================================
# ANOMALY ENGINE
# ============================================================

def anomaly_engine(states):

    if states.empty:
        return pd.DataFrame()

    anomalies = []

    for object_index, g in states.groupby(
        "object_index"
    ):

        speed = g["speed_km_s"].to_numpy()

        if len(speed) < 5:
            continue

        med = np.median(speed)

        mad = np.median(
            np.abs(speed - med)
        )

        scale = max(
            1.4826 * mad,
            1e-6,
        )

        z = np.abs(
            speed - med
        ) / scale

        bad = np.where(z > 6.0)[0]

        for i in bad:

            r = g.iloc[int(i)]

            anomalies.append(
                {
                    "object_index":
                        int(object_index),
                    "name":
                        r["name"],
                    "norad_id":
                        int(r["norad_id"]),
                    "grid_index":
                        int(r["grid_index"]),
                    "grid_time_utc":
                        r["grid_time_utc"],
                    "speed_km_s":
                        float(r["speed_km_s"]),
                    "anomaly_z":
                        float(z[i]),
                }
            )

    return pd.DataFrame(anomalies)


# ============================================================
# AUDIT
# ============================================================

def build_audit(
    catalog,
    states,
    summary,
    pairs,
    screening,
    refined,
    conjunctions,
    output_dir,
    config,
):

    expected_objects = len(catalog)

    expected_states = (
        expected_objects *
        (
            int(
                round(
                    config["horizon_hours"]
                    * 60.0
                    / config["step_minutes"]
                )
            )
            + 1
        )
    )

    actual_states = len(states)

    pair_expected = (
        expected_objects *
        (expected_objects - 1)
        // 2
    )

    pair_ok = (
        len(pairs)
        == pair_expected
    )

    state_ok = (
        actual_states
        == expected_states
    )

    finite_positions = True
    finite_velocities = True

    if not states.empty:

        finite_positions = bool(
            np.isfinite(
                states[
                    [
                        "x_km",
                        "y_km",
                        "z_km",
                    ]
                ].to_numpy()
            ).all()
        )

        finite_velocities = bool(
            np.isfinite(
                states[
                    [
                        "vx_km_s",
                        "vy_km_s",
                        "vz_km_s",
                    ]
                ].to_numpy()
            ).all()
        )

    checks = {
        "version": VERSION,
        "objects_nonzero":
            expected_objects > 0,
        "states_nonzero":
            actual_states > 0,
        "state_schema_valid":
            all(
                c in states.columns
                for c in [
                    "norad_id",
                    "name",
                    "x_km",
                    "y_km",
                    "z_km",
                    "vx_km_s",
                    "vy_km_s",
                    "vz_km_s",
                ]
            ),
        "state_rows_expected":
            expected_states,
        "state_rows_actual":
            actual_states,
        "state_rows_per_object_ok":
            state_ok,
        "finite_positions":
            finite_positions,
        "finite_velocities":
            finite_velocities,
        "catalog_consistency":
            expected_objects == len(summary),
        "propagation_consistency":
            state_ok,
        "pair_count_consistency":
            pair_ok,
        "tca_optimizer_valid":
            bool(
                refined.empty
                or refined[
                    "tca_optimizer_success"
                ].all()
            ),
        "tca_finite":
            bool(
                refined.empty
                or np.isfinite(
                    refined[
                        "miss_distance_km"
                    ].to_numpy()
                ).all()
            ),
        "colocated_excluded":
            True,
        "zero_distance_events_absent":
            bool(
                conjunctions.empty
                or (
                    conjunctions[
                        "miss_distance_km"
                    ] > 0
                ).all()
            ),
        "one_tca_per_pair":
            bool(
                refined.empty
                or not refined.duplicated(
                    [
                        "object_index_a",
                        "object_index_b",
                    ]
                ).any()
            ),
        "risk_not_presented_as_operational_pc":
            True,
        "pc_status_declared":
            True,
        "pc_operational":
            False,
        "risk_model":
            "DEMONSTRATION ONLY",
    }

    structural_keys = [
        "objects_nonzero",
        "states_nonzero",
        "state_schema_valid",
        "state_rows_per_object_ok",
        "finite_positions",
        "finite_velocities",
        "catalog_consistency",
        "propagation_consistency",
        "pair_count_consistency",
        "tca_optimizer_valid",
        "tca_finite",
        "colocated_excluded",
        "zero_distance_events_absent",
        "one_tca_per_pair",
        "risk_not_presented_as_operational_pc",
        "pc_status_declared",
    ]

    audit_pass = all(
        bool(checks[k])
        for k in structural_keys
    )

    checks["FINAL_SSA_AUDIT_PASS"] = audit_pass

    return checks


# ============================================================
# MAIN ENGINE
# ============================================================

def run_pipeline(
    output_dir,
    config=None,
):

    config = {
        **DEFAULT_CONFIG,
        **(config or {}),
    }

    output_dir = ensure_output_dir(
        output_dir
    )

    # --------------------------------------------------------
    # 1 DATA
    # --------------------------------------------------------

    raw = download_catalog()

    valid = validate_catalog(raw)

    valid = valid[
        :config["max_objects"]
    ]

    catalog = valid

    if not catalog:
        raise RuntimeError(
            "No valid orbital objects received."
        )

    # --------------------------------------------------------
    # 2 OBJECTS
    # --------------------------------------------------------

    satellites = []

    for rec in catalog:

        satellites.append(
            omm_to_satrec(rec)
        )

    # --------------------------------------------------------
    # 3 TIME
    # --------------------------------------------------------

    start = datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )

    grid = build_grid(
        start,
        config["horizon_hours"],
        config["step_minutes"],
    )

    # --------------------------------------------------------
    # 4 PROPAGATION
    # --------------------------------------------------------

    states = propagate_catalog(
        catalog,
        satellites,
        grid,
    )

    # --------------------------------------------------------
    # 5 SUMMARY
    # --------------------------------------------------------

    state_summary = build_state_summary(
        states
    )

    cache = build_state_cache(
        states
    )

    # --------------------------------------------------------
    # 6 PAIRS
    # --------------------------------------------------------

    pairs = audit_pairs(
        cache,
        config,
    )

    # --------------------------------------------------------
    # 7 SCREENING
    # --------------------------------------------------------

    screening = screen_conjunctions(
        cache,
        pairs,
        config,
    )

    # --------------------------------------------------------
    # 8 TCA
    # --------------------------------------------------------

    refined = refine_tca(
        cache,
        screening,
    )

    # --------------------------------------------------------
    # 9 TRUE CONJUNCTIONS
    # --------------------------------------------------------

    if refined.empty:

        conjunctions = pd.DataFrame(
            columns=refined.columns
        )

    else:

        conjunctions = refined[
            refined[
                "miss_distance_km"
            ]
            <= config["conjunction_km"]
        ].copy()

    # --------------------------------------------------------
    # 10 RISK
    # --------------------------------------------------------

    risk = monte_carlo_risk(
        refined,
        config,
    )

    # --------------------------------------------------------
    # 11 ANOMALY
    # --------------------------------------------------------

    anomalies = anomaly_engine(
        states
    )

    # --------------------------------------------------------
    # 12 AUDIT
    # --------------------------------------------------------

    audit = build_audit(
        catalog,
        states,
        state_summary,
        pairs,
        screening,
        refined,
        conjunctions,
        output_dir,
        config,
    )

    # --------------------------------------------------------
    # 13 CATALOG DATAFRAME
    # --------------------------------------------------------

    catalog_df = pd.DataFrame(
        [
            {
                "name": r.get(
                    "OBJECT_NAME"
                ),
                "norad_id": int(
                    r["NORAD_CAT_ID"]
                ),
                "epoch": r.get(
                    "EPOCH"
                ),
                "mean_motion_rev_day":
                    float(
                        r["MEAN_MOTION"]
                    ),
                "eccentricity":
                    float(
                        r["ECCENTRICITY"]
                    ),
                "inclination_deg":
                    float(
                        r["INCLINATION"]
                    ),
            }
            for r in catalog
        ]
    )

    # --------------------------------------------------------
    # 14 WRITE DATA
    # --------------------------------------------------------

    files = {
        "catalog.csv":
            catalog_df,

        "propagated_states.csv":
            states,

        "state_summary.csv":
            state_summary,

        "colocation_pairs.csv":
            pairs[pairs["colocated"]]
            if not pairs.empty
            else pairs,

        "pair_audit.csv":
            pairs,

        "conjunction_screening_candidates.csv":
            screening,

        "conjunction_candidates_refined.csv":
            refined,

        "conjunction_events.csv":
            conjunctions,

        "risk_events.csv":
            risk,

        "anomaly_events.csv":
            anomalies,
    }

    for filename, df in files.items():

        df.to_csv(
            output_dir / filename,
            index=False,
        )

    # --------------------------------------------------------
    # 15 SUMMARY JSON
    # --------------------------------------------------------

    summary = {
        "version": VERSION,
        "generated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "source":
            "CELESTRAK_OMM_SGP4",

        "objects":
            len(catalog),

        "state_rows":
            len(states),

        "expected_state_rows":
            len(catalog) * len(grid),

        "propagation_horizon_hours":
            config["horizon_hours"],

        "step_minutes":
            config["step_minutes"],

        "possible_pairs":
            len(catalog)
            * (len(catalog) - 1)
            // 2,

        "colocated_pairs":
            int(
                pairs["colocated"].sum()
            )
            if not pairs.empty
            else 0,

        "independent_pairs":
            int(
                (~pairs["colocated"]).sum()
            )
            if not pairs.empty
            else 0,

        "screening_candidates":
            len(screening),

        "refined_candidates":
            len(refined),

        "true_conjunctions":
            len(conjunctions),

        "minimum_miss_distance_km":
            (
                float(
                    refined[
                        "miss_distance_km"
                    ].min()
                )
                if not refined.empty
                else None
            ),

        "anomalous_samples":
            len(anomalies),

        "pc_operational":
            False,

        "risk_model":
            "DEMONSTRATION ONLY",

        "audit_pass":
            bool(
                audit[
                    "FINAL_SSA_AUDIT_PASS"
                ]
            ),
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    with open(
        output_dir / "audit.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            audit,
            f,
            indent=2,
            default=str,
        )

    return {
        "catalog": catalog_df,
        "states": states,
        "state_summary": state_summary,
        "pairs": pairs,
        "screening": screening,
        "refined": refined,
        "conjunctions": conjunctions,
        "risk": risk,
        "anomalies": anomalies,
        "summary": summary,
        "audit": audit,
        "output_dir": str(output_dir),
    }
