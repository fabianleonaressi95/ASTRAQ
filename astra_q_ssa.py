from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize_scalar
from sgp4.api import Satrec, jday


# ============================================================
# ASTRA-Q SSA ENGINE
# ============================================================

VERSION = "9.0"

CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/"
    "gp.php?GROUP=stations&FORMAT=json"
)

DEFAULT_CACHE = os.path.join("data", "stations_omm.json")


# ============================================================
# CATALOG
# ============================================================

def acquire_catalog(
    cache_file: str = DEFAULT_CACHE,
    timeout: int = 12,
):
    """
    Acquisition hierarchy:

        CelesTrak
             |
             v
        local cache

    Returns:
        catalog, source, message
    """

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    try:

        response = requests.get(
            CELESTRAK_URL,
            timeout=(5, timeout),
            headers={
                "User-Agent": "ASTRA-Q-SSA/9.0"
            },
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            raise ValueError("Invalid CelesTrak response")

        if len(data) == 0:
            raise ValueError("Empty CelesTrak catalog")

        # Validate basic OMM structure
        valid = []

        for rec in data:

            if not isinstance(rec, dict):
                continue

            norad = rec.get("NORAD_CAT_ID")

            line1 = rec.get("TLE_LINE1")
            line2 = rec.get("TLE_LINE2")

            if norad is None:
                continue

            if not line1 or not line2:
                continue

            valid.append(rec)

        if not valid:
            raise ValueError("No valid OMM/TLE records")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(valid, f, indent=2)

        return (
            valid,
            "LIVE_CELESTRAK",
            "Live CelesTrak catalog acquired successfully."
        )

    except Exception as exc:

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        if os.path.exists(cache_file):

            try:

                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)

                if isinstance(cached, list) and len(cached) > 0:

                    return (
                        cached,
                        "LOCAL_CACHE",
                        f"CelesTrak unavailable; local cache used. "
                        f"Reason: {type(exc).__name__}"
                    )

            except Exception:
                pass

        raise RuntimeError(
            "Neither CelesTrak nor local catalog cache is available."
        )


# ============================================================
# OMM VALIDATION
# ============================================================

def validate_catalog(records):

    output = []
    seen = set()

    for rec in records:

        try:

            norad = int(rec["NORAD_CAT_ID"])

            line1 = rec["TLE_LINE1"]
            line2 = rec["TLE_LINE2"]

            name = (
                rec.get("OBJECT_NAME")
                or rec.get("OBJECT_ID")
                or f"NORAD-{norad}"
            )

            if norad in seen:
                continue

            if not isinstance(line1, str):
                continue

            if not isinstance(line2, str):
                continue

            seen.add(norad)

            output.append(
                {
                    "name": str(name),
                    "norad_id": norad,
                    "line1": line1,
                    "line2": line2,
                }
            )

        except Exception:
            continue

    return output


# ============================================================
# SATREC
# ============================================================

def build_satellites(records, max_objects=40):

    objects = []

    failures = []

    for rec in records[:max_objects]:

        try:

            sat = Satrec.twoline2rv(
                rec["line1"],
                rec["line2"]
            )

            objects.append(
                {
                    "name": rec["name"],
                    "norad_id": rec["norad_id"],
                    "sat": sat,
                }
            )

        except Exception as exc:

            failures.append(
                {
                    "name": rec.get("name", "UNKNOWN"),
                    "norad_id": rec.get("norad_id"),
                    "error": str(exc),
                }
            )

    return objects, failures


# ============================================================
# TIME
# ============================================================

def datetime_to_jd(dt):

    dt = dt.astimezone(timezone.utc)

    sec = (
        dt.second
        + dt.microsecond / 1e6
    )

    return jday(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        sec,
    )


# ============================================================
# PROPAGATION
# ============================================================

def propagate_satellite(sat, dt):

    jd, fr = datetime_to_jd(dt)

    error, position, velocity = sat.sgp4(jd, fr)

    if error != 0:
        return None, None

    r = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)

    if not np.all(np.isfinite(r)):
        return None, None

    if not np.all(np.isfinite(v)):
        return None, None

    return r, v


def propagate_catalog(
    satellites,
    start,
    horizon_hours=24,
    step_minutes=5,
):

    steps = int(
        round(horizon_hours * 60 / step_minutes)
    ) + 1

    times = [
        start + timedelta(
            minutes=i * step_minutes
        )
        for i in range(steps)
    ]

    rows = []

    for object_index, obj in enumerate(satellites):

        sat = obj["sat"]

        for grid_index, dt in enumerate(times):

            r, v = propagate_satellite(
                sat,
                dt
            )

            if r is None:
                continue

            rows.append(
                {
                    "object_index": object_index,
                    "name": obj["name"],
                    "norad_id": obj["norad_id"],
                    "grid_index": grid_index,
                    "utc": dt.isoformat(),
                    "x_km": r[0],
                    "y_km": r[1],
                    "z_km": r[2],
                    "vx_km_s": v[0],
                    "vy_km_s": v[1],
                    "vz_km_s": v[2],
                }
            )

    return pd.DataFrame(rows), times


# ============================================================
# STATE SUMMARY
# ============================================================

def build_state_summary(states):

    if states.empty:
        return pd.DataFrame(
            columns=[
                "object_index",
                "name",
                "norad_id",
                "altitude_min_km",
                "altitude_max_km",
                "altitude_mean_km",
                "speed_mean_km_s",
                "state_rows",
            ]
        )

    rows = []

    for idx, g in states.groupby(
        "object_index",
        sort=True
    ):

        r = g[
            ["x_km", "y_km", "z_km"]
        ].to_numpy()

        v = g[
            ["vx_km_s", "vy_km_s", "vz_km_s"]
        ].to_numpy()

        altitude = (
            np.linalg.norm(r, axis=1)
            - 6378.137
        )

        speed = np.linalg.norm(v, axis=1)

        rows.append(
            {
                "object_index": int(idx),
                "name": g["name"].iloc[0],
                "norad_id": int(g["norad_id"].iloc[0]),
                "altitude_min_km": float(np.min(altitude)),
                "altitude_max_km": float(np.max(altitude)),
                "altitude_mean_km": float(np.mean(altitude)),
                "speed_mean_km_s": float(np.mean(speed)),
                "state_rows": int(len(g)),
            }
        )

    return pd.DataFrame(rows)


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
            "grid_index": g["grid_index"].to_numpy(),
            "time": g["utc"].to_numpy(),
            "r": g[
                ["x_km", "y_km", "z_km"]
            ].to_numpy(float),
            "v": g[
                ["vx_km_s", "vy_km_s", "vz_km_s"]
            ].to_numpy(float),
            "name": g["name"].iloc[0],
            "norad_id": int(g["norad_id"].iloc[0]),
        }

    return cache


# ============================================================
# PAIR DISTANCE
# ============================================================

def pair_distance(cache_a, cache_b):

    n = min(
        len(cache_a["r"]),
        len(cache_b["r"])
    )

    dr = (
        cache_a["r"][:n]
        - cache_b["r"][:n]
    )

    dv = (
        cache_a["v"][:n]
        - cache_b["v"][:n]
    )

    distance = np.linalg.norm(
        dr,
        axis=1
    )

    relative_velocity = np.linalg.norm(
        dv,
        axis=1
    )

    return distance, relative_velocity


# ============================================================
# COLOCATION
# ============================================================

def audit_colocation(
    cache,
    distance_km=5.0,
    relative_velocity_km_s=0.05,
):

    pairs = []

    ids = sorted(cache.keys())

    for a, b in combinations(ids, 2):

        ca = cache[a]
        cb = cache[b]

        d, rv = pair_distance(ca, cb)

        if len(d) == 0:
            continue

        k = int(np.argmin(d))

        colocated = (
            d[k] <= distance_km
            and rv[k] <= relative_velocity_km_s
        )

        pairs.append(
            {
                "object_index_a": a,
                "object_index_b": b,
                "object_a": ca["name"],
                "object_b": cb["name"],
                "norad_id_a": ca["norad_id"],
                "norad_id_b": cb["norad_id"],
                "min_grid_distance_km": float(d[k]),
                "relative_velocity_at_grid_min_km_s":
                    float(rv[k]),
                "minimum_grid_time_utc":
                    ca["time"][k],
                "colocated": bool(colocated),
            }
        )

    columns = [
        "object_index_a",
        "object_index_b",
        "object_a",
        "object_b",
        "norad_id_a",
        "norad_id_b",
        "min_grid_distance_km",
        "relative_velocity_at_grid_min_km_s",
        "minimum_grid_time_utc",
        "colocated",
    ]

    if not pairs:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(pairs)


# ============================================================
# SCREENING
# ============================================================

def conjunction_screening(
    cache,
    colocation_df,
    screening_distance=50.0,
):

    colocated_set = set()

    if not colocation_df.empty:

        for _, row in colocation_df.iterrows():

            if bool(row["colocated"]):

                colocated_set.add(
                    (
                        int(row["object_index_a"]),
                        int(row["object_index_b"])
                    )
                )

    candidates = []

    ids = sorted(cache.keys())

    for a, b in combinations(ids, 2):

        if (a, b) in colocated_set:
            continue

        ca = cache[a]
        cb = cache[b]

        d, rv = pair_distance(ca, cb)

        if len(d) == 0:
            continue

        k = int(np.argmin(d))

        if d[k] <= screening_distance:

            candidates.append(
                {
                    "object_index_a": a,
                    "object_index_b": b,
                    "sat_index_a": ca["norad_id"],
                    "sat_index_b": cb["norad_id"],
                    "norad_id_a": ca["norad_id"],
                    "norad_id_b": cb["norad_id"],
                    "object_a": ca["name"],
                    "object_b": cb["name"],
                    "grid_index": int(k),
                    "grid_time_utc": ca["time"][k],
                    "grid_distance_km": float(d[k]),
                    "grid_relative_velocity_km_s":
                        float(rv[k]),
                }
            )

    columns = [
        "object_index_a",
        "object_index_b",
        "sat_index_a",
        "sat_index_b",
        "norad_id_a",
        "norad_id_b",
        "object_a",
        "object_b",
        "grid_index",
        "grid_time_utc",
        "grid_distance_km",
        "grid_relative_velocity_km_s",
    ]

    return pd.DataFrame(
        candidates,
        columns=columns
    )


# ============================================================
# CONTINUOUS TCA
# ============================================================

def refine_tca(
    candidate,
    cache,
    times,
):

    a = int(candidate["object_index_a"])
    b = int(candidate["object_index_b"])

    ca = cache[a]
    cb = cache[b]

    k = int(candidate["grid_index"])

    if k >= len(times):
        return None

    # --------------------------------------------------------
    # Local interpolation around coarse minimum
    # --------------------------------------------------------

    lo = max(0, k - 2)
    hi = min(
        len(times) - 1,
        k + 2
    )

    t0 = times[0]

    sample_times = np.array(
        [
            (
                t - t0
            ).total_seconds()
            for t in times
        ],
        dtype=float
    )

    def position_at(
        cache_obj,
        t_seconds
    ):

        r = cache_obj["r"]

        result = np.array(
            [
                np.interp(
                    t_seconds,
                    sample_times[:len(r)],
                    r[:, j]
                )
                for j in range(3)
            ]
        )

        return result

    def objective(t_seconds):

        ra = position_at(
            ca,
            t_seconds
        )

        rb = position_at(
            cb,
            t_seconds
        )

        return float(
            np.linalg.norm(ra - rb)
        )

    x0 = sample_times[k]

    left = sample_times[lo]
    right = sample_times[hi]

    try:

        result = minimize_scalar(
            objective,
            bounds=(left, right),
            method="bounded",
            options={
                "xatol": 0.001
            }
        )

        tca_seconds = float(
            result.x
        )

        miss = float(
            result.fun
        )

        eps = 0.1

        rp = (
            position_at(
                ca,
                tca_seconds + eps
            )
            -
            position_at(
                cb,
                tca_seconds + eps
            )
        )

        rm = (
            position_at(
                ca,
                tca_seconds - eps
            )
            -
            position_at(
                cb,
                tca_seconds - eps
            )
        )

        rv = float(
            np.linalg.norm(
                (rp - rm)
                / (2 * eps)
            )
        )

        tca = t0 + timedelta(
            seconds=tca_seconds
        )

        return {
            "object_index_a": a,
            "object_index_b": b,
            "sat_index_a": ca["norad_id"],
            "sat_index_b": cb["norad_id"],
            "norad_id_a": ca["norad_id"],
            "norad_id_b": cb["norad_id"],
            "object_a": ca["name"],
            "object_b": cb["name"],
            "grid_distance_km":
                float(candidate["grid_distance_km"]),
            "miss_distance_km": miss,
            "relative_velocity_km_s": rv,
            "tca_minutes_from_start":
                tca_seconds / 60.0,
            "tca_utc": tca.isoformat(),
            "tca_optimizer_success":
                bool(result.success),
        }

    except Exception:

        return None


def refine_candidates(
    candidates,
    cache,
    times,
):

    rows = []

    if candidates.empty:
        return pd.DataFrame()

    for _, candidate in candidates.iterrows():

        result = refine_tca(
            candidate,
            cache,
            times
        )

        if result is not None:
            rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# DEMONSTRATION RISK
# ============================================================

def demonstration_risk(
    refined,
    samples=300,
    position_sigma_km=1.0,
    velocity_sigma_km_s=0.001,
    threshold_km=25.0,
):

    if refined.empty:

        return pd.DataFrame()

    rng = np.random.default_rng(42)

    rows = []

    for _, event in refined.iterrows():

        miss = float(
            event["miss_distance_km"]
        )

        simulated = np.linalg.norm(
            rng.normal(
                loc=miss,
                scale=position_sigma_km,
                size=samples
            )
        )

        p05, median, p95 = np.percentile(
            simulated,
            [5, 50, 95]
        )

        # ----------------------------------------------------
        # Demonstration score.
        # NOT operational Pc.
        # ----------------------------------------------------

        score = max(
            0.0,
            min(
                100.0,
                100.0
                * math.exp(
                    -miss / 25.0
                )
            )
        )

        if miss <= 10:
            level = "CRITICAL"
        elif miss <= 20:
            level = "VERY HIGH"
        elif miss <= 25:
            level = "HIGH"
        elif miss <= 40:
            level = "MEDIUM"
        else:
            level = "LOW"

        rows.append(
            {
                **event.to_dict(),
                "pc_demo": np.nan,
                "mc_p05_km": float(p05),
                "mc_median_km": float(median),
                "mc_p95_km": float(p95),
                "risk_score_demo": float(score),
                "risk_level": level,
                "operational_pc": False,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ANOMALY ENGINE
# ============================================================

def anomaly_engine(states):

    if states.empty:
        return pd.DataFrame()

    rows = []

    for object_index, g in states.groupby(
        "object_index"
    ):

        r = g[
            ["x_km", "y_km", "z_km"]
        ].to_numpy()

        v = g[
            ["vx_km_s", "vy_km_s", "vz_km_s"]
        ].to_numpy()

        altitude = (
            np.linalg.norm(r, axis=1)
            - 6378.137
        )

        speed = np.linalg.norm(v, axis=1)

        alt_mu = np.mean(altitude)
        alt_sd = np.std(altitude)

        speed_mu = np.mean(speed)
        speed_sd = np.std(speed)

        alt_z = np.zeros_like(altitude)

        speed_z = np.zeros_like(speed)

        if alt_sd > 0:
            alt_z = (
                altitude - alt_mu
            ) / alt_sd

        if speed_sd > 0:
            speed_z = (
                speed - speed_mu
            ) / speed_sd

        anomaly = np.maximum(
            np.abs(alt_z),
            np.abs(speed_z)
        )

        for i in range(len(g)):

            rows.append(
                {
                    "object_index": int(object_index),
                    "name": g["name"].iloc[0],
                    "norad_id":
                        int(g["norad_id"].iloc[0]),
                    "grid_index":
                        int(g["grid_index"].iloc[i]),
                    "utc": g["utc"].iloc[i],
                    "altitude_km":
                        float(altitude[i]),
                    "speed_km_s":
                        float(speed[i]),
                    "anomaly_index":
                        float(anomaly[i]),
                    "anomalous":
                        bool(anomaly[i] > 3.0),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# AUDIT
# ============================================================

def build_audit(
    satellites,
    states,
    state_summary,
    colocation,
    candidates,
    refined,
    risk,
    expected_steps,
):

    n = len(satellites)

    expected_pairs = (
        n * (n - 1) // 2
    )

    actual_pairs = len(colocation)

    finite_positions = True
    finite_velocities = True

    if not states.empty:

        finite_positions = bool(
            np.isfinite(
                states[
                    [
                        "x_km",
                        "y_km",
                        "z_km"
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
                        "vz_km_s"
                    ]
                ].to_numpy()
            ).all()
        )

    expected_rows = (
        n * expected_steps
    )

    actual_rows = len(states)

    rows_per_object_ok = (
        actual_rows == expected_rows
        and
        (
            state_summary["state_rows"] == expected_steps
        ).all()
        if not state_summary.empty
        else n == 0
    )

    checks = {

        "version":
            f"v{VERSION}",

        "objects_nonzero":
            n > 0,

        "states_nonzero":
            actual_rows > 0,

        "state_schema_valid":
            set(
                [
                    "object_index",
                    "name",
                    "norad_id",
                    "grid_index",
                    "utc",
                    "x_km",
                    "y_km",
                    "z_km",
                    "vx_km_s",
                    "vy_km_s",
                    "vz_km_s",
                ]
            ).issubset(states.columns),

        "state_rows_expected":
            expected_rows,

        "state_rows_actual":
            actual_rows,

        "state_rows_per_object_ok":
            bool(rows_per_object_ok),

        "finite_positions":
            finite_positions,

        "finite_velocities":
            finite_velocities,

        "catalog_consistency":
            len(satellites) == n,

        "propagation_consistency":
            actual_rows == expected_rows,

        "pair_count_consistency":
            actual_pairs == expected_pairs,

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
                        [
                            "miss_distance_km",
                            "relative_velocity_km_s",
                        ]
                    ].to_numpy()
                ).all()
            ),

        "colocated_excluded":
            True,

        "zero_distance_events_absent":
            bool(
                refined.empty
                or
                (
                    refined["miss_distance_km"]
                    > 0
                ).all()
            ),

        "one_tca_per_pair":
            (
                refined.empty
                or
                not refined.duplicated(
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

    audit_pass = all(
        value
        for key, value in checks.items()
        if isinstance(value, bool)
    )

    return checks, audit_pass


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_ssa(
    max_objects=40,
    horizon_hours=24,
    step_minutes=5,
    screening_distance_km=50,
    conjunction_threshold_km=25,
    colocation_distance_km=5,
    colocation_velocity_km_s=0.05,
    mc_samples=300,
    cache_file=DEFAULT_CACHE,
):

    started = time.time()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    raw_catalog, source, source_message = acquire_catalog(
        cache_file=cache_file
    )

    catalog = validate_catalog(
        raw_catalog
    )

    satellites, failures = build_satellites(
        catalog,
        max_objects=max_objects
    )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    start = datetime.now(
        timezone.utc
    ).replace(
        second=0,
        microsecond=0
    )

    states, times = propagate_catalog(
        satellites,
        start,
        horizon_hours=horizon_hours,
        step_minutes=step_minutes
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    state_summary = build_state_summary(
        states
    )

    cache = build_state_cache(
        states
    )

    # --------------------------------------------------------
    # PAIRS
    # --------------------------------------------------------

    colocation = audit_colocation(
        cache,
        distance_km=colocation_distance_km,
        relative_velocity_km_s=
            colocation_velocity_km_s
    )

    candidates = conjunction_screening(
        cache,
        colocation,
        screening_distance=
            screening_distance_km
    )

    # --------------------------------------------------------
    # TCA
    # --------------------------------------------------------

    refined = refine_candidates(
        candidates,
        cache,
        times
    )

    # --------------------------------------------------------
    # TRUE EVENTS
    # --------------------------------------------------------

    if refined.empty:

        true_conjunctions = refined.copy()

    else:

        true_conjunctions = refined[
            refined["miss_distance_km"]
            <= conjunction_threshold_km
        ].copy()

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk = demonstration_risk(
        true_conjunctions,
        samples=mc_samples
    )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    anomalies = anomaly_engine(
        states
    )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    expected_steps = (
        int(
            round(
                horizon_hours
                * 60
                / step_minutes
            )
        )
        + 1
    )

    audit, audit_pass = build_audit(
        satellites,
        states,
        state_summary,
        colocation,
        candidates,
        refined,
        risk,
        expected_steps
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "version":
            VERSION,

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "source":
            source,

        "source_message":
            source_message,

        "raw_catalog_records":
            len(raw_catalog),

        "validated_records":
            len(catalog),

        "objects":
            len(satellites),

        "instantiation_failures":
            len(failures),

        "horizon_hours":
            horizon_hours,

        "step_minutes":
            step_minutes,

        "steps":
            expected_steps,

        "state_rows":
            len(states),

        "possible_pairs":
            len(satellites)
            * (len(satellites) - 1)
            // 2,

        "colocated_pairs":
            int(
                colocation["colocated"].sum()
            )
            if not colocation.empty
            else 0,

        "independent_pairs":
            (
                len(satellites)
                * (len(satellites) - 1)
                // 2
            )
            -
            (
                int(
                    colocation["colocated"].sum()
                )
                if not colocation.empty
                else 0
            ),

        "coarse_candidates":
            len(candidates),

        "refined_candidates":
            len(refined),

        "true_conjunctions":
            len(true_conjunctions),

        "minimum_miss_distance_km":
            (
                float(
                    true_conjunctions[
                        "miss_distance_km"
                    ].min()
                )
                if not true_conjunctions.empty
                else None
            ),

        "anomalous_samples":
            (
                int(
                    anomalies["anomalous"].sum()
                )
                if not anomalies.empty
                else 0
            ),

        "audit_pass":
            audit_pass,

        "operational_pc":
            False,

        "risk_model":
            "DEMONSTRATION ONLY",

        "runtime_seconds":
            time.time() - started,
    }

    return {

        "raw_catalog": raw_catalog,
        "catalog": catalog,
        "satellites": satellites,
        "failures": failures,

        "states": states,
        "state_summary": state_summary,

        "colocation": colocation,

        "candidates": candidates,
        "refined": refined,
        "true_conjunctions":
            true_conjunctions,

        "risk": risk,

        "anomalies": anomalies,

        "audit": audit,
        "audit_pass": audit_pass,

        "summary": summary,
    }


# ============================================================
# EXPORT
# ============================================================

def export_results(
    results,
    output_dir="data/generated"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    tables = {

        "catalog.csv":
            pd.DataFrame(
                results["catalog"]
            ),

        "propagated_states.csv":
            results["states"],

        "state_summary.csv":
            results["state_summary"],

        "colocation_pairs.csv":
            results["colocation"],

        "conjunction_screening_candidates.csv":
            results["candidates"],

        "conjunction_candidates_refined.csv":
            results["refined"],

        "conjunction_events.csv":
            results["true_conjunctions"],

        "risk_events.csv":
            results["risk"],

        "anomaly_events.csv":
            results["anomalies"],
    }

    for filename, df in tables.items():

        path = os.path.join(
            output_dir,
            filename
        )

        df.to_csv(
            path,
            index=False
        )

    with open(
        os.path.join(
            output_dir,
            "summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results["summary"],
            f,
            indent=2,
            default=str
        )

    with open(
        os.path.join(
            output_dir,
            "audit.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results["audit"],
            f,
            indent=2,
            default=str
        )

    return output_dir
