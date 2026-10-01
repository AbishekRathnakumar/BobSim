"""Strict vectorized zero-camber MF5.2 pure-lateral force.

Equation source: BobLib/.../MF52/PureSlip/FyPureEval.mo. This narrow adapter
retains signed coefficients and rejects unsupported loads/slip, without a mu floor.
"""

from __future__ import annotations

import numpy as np


class LateralForce:
    """Precompute load-dependent terms for repeated constant-slip evaluations."""

    def __init__(self, tire, loads):
        p = self.p = tire
        z = np.asarray(loads, dtype=float)
        if z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError("Loads must be finite and nonempty")
        if np.any(z < p["FZMIN"]) or np.any(z > p["FZMAX"]):
            raise ValueError("Load outside fitted tire domain; no clipping or extrapolation")
        if not p["CAMMIN"] <= 0 <= p["CAMMAX"]:
            raise ValueError("Zero camber is outside fitted tire domain")
        dfz = (z - p["FNOMIN"] * p["LFZO"]) / (p["FNOMIN"] * p["LFZO"])
        self.c = p["PCY1"] * p["LCY"]
        self.d = (p["PDY1"] + p["PDY2"] * dfz) * p["LMUY"] * z
        stiffness = p["PKY1"] * p["FNOMIN"] * np.sin(2 * np.arctan(z / (p["PKY2"] * p["FNOMIN"] * p["LFZO"])))
        stiffness *= p["LFZO"] * p["LKY"]
        self.b = stiffness / (self.c * self.d + 1e-8)
        self.sh = (p["PHY1"] + p["PHY2"] * dfz) * p["LHY"]
        self.sv = z * (p["PVY1"] + p["PVY2"] * dfz) * p["LVY"] * p["LMUY"]
        self.e_base = (p["PEY1"] + p["PEY2"] * dfz) * p["LEY"]

    def __call__(self, alpha):
        if (
            not np.all(np.isfinite(alpha))
            or np.any(np.asarray(alpha) < self.p["ALPMIN"])
            or np.any(np.asarray(alpha) > self.p["ALPMAX"])
        ):
            raise ValueError("Slip angle outside fitted tire domain")
        slip = alpha + self.sh
        e = np.minimum(self.e_base * (1 - self.p["PEY3"] * np.sign(slip)), 1)
        x = self.b * slip
        return self.d * np.sin(self.c * np.arctan(x - e * (x - np.arctan(x)))) + self.sv
