# ASTRA-Q SSA

## Space Situational Awareness & Orbital Intelligence

ASTRA-Q SSA is a technology demonstrator for dynamic orbital
intelligence.

The system ingests orbital data, validates the catalog, propagates
objects through SGP4, performs pairwise screening, excludes
co-located/docked objects, refines conjunction candidates using
continuous TCA optimization, and exposes uncertainty and anomaly
layers.

## Demonstrator

Current demonstration:

- 22 orbital objects
- 24-hour propagation horizon
- 289 propagation epochs
- 6,358 propagated state records
- 231 possible object pairs
- co-location exclusion
- conjunction screening
- continuous TCA refinement
- uncertainty envelope
- anomaly monitoring
- structural SSA audit

## Important limitation

This demonstrator does NOT calculate an operational probability
of collision (Pc).

Monte Carlo uncertainty envelopes and risk scores are
demonstration-only and must not be used for operational
collision avoidance.

## Architecture

CelesTrak OMM
        ↓
Catalog Validation
        ↓
SGP4 Propagation
        ↓
State Engine
        ↓
Relationship / Co-location Engine
        ↓
Conjunction Screening
        ↓
Continuous TCA
        ↓
Uncertainty Layer
        ↓
Anomaly Engine
        ↓
SSA Audit
        ↓
Streamlit Intelligence Dashboard

## Positioning

ASTRA-Q is designed as a:

> Dynamic Intelligence & Monitoring Platform

Space Situational Awareness is the initial vertical demonstration.

The architecture is intended to be reusable for other
dynamic monitoring applications.

## Deployment

The application is designed for Streamlit Community Cloud.
