# ASTRA-Q

## Space Situational Awareness & Universal Scaling Law Analysis

ASTRA-Q is a multi-purpose platform combining:

1. **Space Situational Awareness (SSA)** - Technology demonstrator for dynamic orbital intelligence
2. **Universal Nonlinear Scaling Law Analysis** - Analysis framework for perturbed renormalization group flows

---

## Space Situational Awareness (SSA)

The SSA system ingests orbital data, validates the catalog, propagates
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

---

## Universal Nonlinear Scaling Law Analysis

### Overview

This module implements the analysis framework for the universal scaling law discovered in Audit V4.2-V4.3:

**I_nonadd(ε,ω,φ₀) = ε² K₂(ω,φ₀) + O(ε³)**

where:
- ε: Perturbation amplitude
- ω: Frequency of modulation
- φ₀: Initial phase
- K₂: Universal kernel of nonlinear response
- I_nonadd: Non-additive observable

### Key Results from Audit V4.2-V4.3

- **Scaling exponent**: p = 1.9996 ± 0.0004 (theoretical: p = 2.0)
- **Universal kernel**: K₂ ≈ 0.1543
- **Linear coefficient**: c₁ ≈ 0 (negligible)
- **Perturbative regime**: Valid for ε < 0.03

### Analysis Scripts

1. **scaling_law_verification.py** - Verifies the quadratic scaling law I ∝ ε^p with p ≈ 2
2. **kernel_K2_analysis.py** - Analyzes the universal kernel K₂(ω,φ₀) and its frequency dependence
3. **perturbation_expansion.py** - Analyzes the perturbation expansion I = c₁ε + c₂ε² + c₃ε³ + ...
4. **universal_scaling_law_demo.py** - Comprehensive demonstration of the complete analysis pipeline

### Usage

```bash
# Run the comprehensive demonstration
python universal_scaling_law_demo.py

# Run individual analysis modules
python scaling_law_verification.py
python kernel_K2_analysis.py
python perturbation_expansion.py
```

### Scientific Significance

The analysis separates two distinct hypotheses:

1. **Scaling universality** (p ≈ 2): Strongly supported by numerical data
2. **Golden-ratio frequency selection** (ω = 2π/ln φ): Not yet statistically supported

This represents a theoretically solid formulation that establishes the quadratic scaling as a robust numerical result while keeping frequency dependence as an open question for experimental validation.

## Deployment

The application is designed for Streamlit Community Cloud.
