"""
ASTRA-Q Universal Nonlinear Scaling Law
=======================================

Constitutive law discovered in Audit V4.2-V4.3:

    I_nonadd(eps, omega, phi0)
        = eps^2 K2(omega, phi0; state)
        + eps^3 K3(...)
        + O(eps^4)

The quadratic exponent is an empirical/theoretical result of the
perturbative audit. Golden-ratio frequency selection remains a
separate hypothesis and must not be hard-coded as universal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np


PHI = (1.0 + np.sqrt(5.0)) / 2.0

# Theoretical candidate, NOT a fitted universal constant.
OMEGA_PHI = 2.0 * np.pi / np.log(PHI)


@dataclass
class ScalingLawConfig:

    # Perturbation amplitude
    epsilon: float = 0.01

    # Modulation frequency
    omega: float = OMEGA_PHI

    # Initial phase
    phi0: float = 0.0

    # Leading-order exponent
    p: float = 2.0

    # Optional measured/fitted kernel
    K2: Optional[float] = None

    # Higher-order coefficients
    K3: float = 0.0
    K4: float = 0.0

    # Perturbative-domain threshold
    epsilon_max_perturbative: float = 0.03


@dataclass
class ScalingLawResult:

    epsilon: float
    omega: float
    phi0: float

    K2: float
    K3: float
    K4: float

    I_quadratic: float
    I_nonlinear: float

    relative_nonlinearity: float

    perturbative: bool

    metadata: Dict[str, Any]


class UniversalScalingLaw:

    """
    ASTRA-Q constitutive scaling model.
    """

    def __init__(
        self,
        K2: float,
        K3: float = 0.0,
        K4: float = 0.0,
        epsilon_max_perturbative: float = 0.03,
    ):

        self.K2 = float(K2)
        self.K3 = float(K3)
        self.K4 = float(K4)

        self.epsilon_max_perturbative = float(
            epsilon_max_perturbative
        )

    # ---------------------------------------------------------
    # Core law
    # ---------------------------------------------------------

    def evaluate(
        self,
        epsilon: float,
        omega: float = OMEGA_PHI,
        phi0: float = 0.0,
    ) -> ScalingLawResult:

        eps = float(epsilon)

        I2 = self.K2 * eps**2

        I3 = self.K3 * eps**3
        I4 = self.K4 * eps**4

        Inl = I2 + I3 + I4

        if abs(I2) > 0:
            relative_nonlinearity = abs(
                Inl - I2
            ) / abs(I2)

        else:
            relative_nonlinearity = 0.0

        return ScalingLawResult(
            epsilon=eps,
            omega=float(omega),
            phi0=float(phi0),

            K2=self.K2,
            K3=self.K3,
            K4=self.K4,

            I_quadratic=I2,
            I_nonlinear=Inl,

            relative_nonlinearity=relative_nonlinearity,

            perturbative=(
                abs(eps)
                <= self.epsilon_max_perturbative
            ),

            metadata={
                "phi": PHI,
                "omega_phi": OMEGA_PHI,
                "law": "I = eps^2 K2 + eps^3 K3 + eps^4 K4",
            },
        )

    # ---------------------------------------------------------
    # Leading-order prediction
    # ---------------------------------------------------------

    def predict(
        self,
        epsilon,
        omega=OMEGA_PHI,
        phi0=0.0,
    ):

        return self.K2 * np.asarray(epsilon)**2

    # ---------------------------------------------------------
    # Scaling exponent
    # ---------------------------------------------------------

    def effective_exponent(
        self,
        eps1: float,
        eps2: float,
    ):

        I1 = abs(self.predict(eps1))
        I2 = abs(self.predict(eps2))

        if I1 <= 0 or I2 <= 0:
            return np.nan

        return np.log(I2 / I1) / np.log(
            abs(eps2 / eps1)
        )

    # ---------------------------------------------------------
    # Perturbative validity
    # ---------------------------------------------------------

    def is_perturbative(self, epsilon):

        return (
            abs(epsilon)
            <= self.epsilon_max_perturbative
        )

    # ---------------------------------------------------------
    # Golden-ratio diagnostic
    # ---------------------------------------------------------

    def golden_frequency_offset(self, omega):

        return float(
            omega - OMEGA_PHI
        )

    # ---------------------------------------------------------
    # Serialize
    # ---------------------------------------------------------

    def to_dict(self):

        return {
            "K2": self.K2,
            "K3": self.K3,
            "K4": self.K4,
            "epsilon_max_perturbative":
                self.epsilon_max_perturbative,
            "phi": PHI,
            "omega_phi": OMEGA_PHI,
        }


def build_audit_v43_law():

    """
    Current numerical audit calibration.

    K2 ≈ 0.1543

    IMPORTANT:
    this is an empirical calibration, not yet a fundamental
    universal constant.
    """

    return UniversalScalingLaw(
        K2=0.154257,
        K3=0.0,
        K4=0.0,
        epsilon_max_perturbative=0.03,
    )
