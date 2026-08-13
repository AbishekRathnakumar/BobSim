"""Reduced double-wishbone force transmission from instantaneous kinematics.

The detailed BobLib suspension resolves every rigid arm, upright, pushrod,
bellcrank, spring, damper, and stabilizer-bar joint.  ``dyn_py`` keeps the
spring/bar path as an equivalent wheel rate, but represents the rigid
wishbone/upright path with one instantaneous link per force plane.

For a contact patch whose constrained jounce motion is
``dr_cp/dz = [dx/dz, dy/dz, 1]``, virtual work gives the reciprocal link-force
coefficients

``Fz_geo/Fx = -dx/dz`` and ``Fz_geo/Fy = -dy/dz``.

Those are precisely the side-view and front-view instantaneous swing-arm line
slopes, without relying on an inaccurate 2D projection of swept 3D arm axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from _0_Utils.kin_py.kinematics import CornerKinematics


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _AxleInstantLinkCurve:
    """Jounce-indexed instantaneous-link coefficients for one left corner."""

    jounce_m: tuple[float, ...]
    longitudinal: tuple[float, ...]
    lateral_left: tuple[float, ...]

    def at(self, jounce_m: float) -> tuple[float, float]:
        grid = np.asarray(self.jounce_m, dtype=float)
        return (
            float(np.interp(jounce_m, grid, self.longitudinal)),
            float(np.interp(jounce_m, grid, self.lateral_left)),
        )


@dataclass(frozen=True)
class CornerInstantLink:
    """Nominal instantaneous link slopes for one suspension corner."""

    longitudinal_jacking_coefficient: float
    lateral_jacking_coefficient: float

    def geometric_vertical_force(self, fx_n: float, fy_n: float) -> float:
        """Return vertical chassis force transmitted by horizontal tire force."""

        return (
            self.longitudinal_jacking_coefficient * float(fx_n)
            + self.lateral_jacking_coefficient * float(fy_n)
        )


@dataclass(frozen=True)
class DoubleWishboneInstantLinks:
    """Instantaneous force links ordered FL, FR, RL, RR."""

    corners: tuple[
        CornerInstantLink,
        CornerInstantLink,
        CornerInstantLink,
        CornerInstantLink,
    ]

    @property
    def coefficient_matrix(self) -> FloatArray:
        """Return rows ``[Fz/Fx, Fz/Fy]`` in FL, FR, RL, RR order."""

        return np.asarray(
            [
                (
                    corner.longitudinal_jacking_coefficient,
                    corner.lateral_jacking_coefficient,
                )
                for corner in self.corners
            ],
            dtype=float,
        )

    def geometric_vertical_forces(
        self,
        fx_n: ArrayLike,
        fy_n: ArrayLike,
    ) -> FloatArray:
        """Map four horizontal contact forces to four geometric vertical forces."""

        fx = np.asarray(fx_n, dtype=float)
        fy = np.asarray(fy_n, dtype=float)
        if fx.shape != (4,) or fy.shape != (4,):
            raise ValueError("Instant-link force inputs must be four-vectors ordered FL, FR, RL, RR.")
        coefficients = self.coefficient_matrix
        return coefficients[:, 0] * fx + coefficients[:, 1] * fy

    @property
    def axle_longitudinal_coefficients(self) -> tuple[float, float]:
        """Return symmetric front and rear longitudinal jacking coefficients."""

        coefficients = self.coefficient_matrix
        return float(np.mean(coefficients[:2, 0])), float(np.mean(coefficients[2:, 0]))

    @property
    def axle_net_lateral_coefficients(self) -> tuple[float, float]:
        """Return net axle heave jacking under equal left/right lateral force."""

        coefficients = self.coefficient_matrix
        return float(np.mean(coefficients[:2, 1])), float(np.mean(coefficients[2:, 1]))


@dataclass(frozen=True)
class DoubleWishboneGeometry:
    """Hardpoint-derived dynamic instant-link curves for front and rear axles."""

    front: _AxleInstantLinkCurve
    rear: _AxleInstantLinkCurve

    @classmethod
    def from_vehicle(
        cls,
        vehicle: Mapping[str, Any],
        *,
        travel_limit_m: float = 0.06,
        sample_count: int = 49,
    ) -> DoubleWishboneGeometry:
        """Solve wishbone constraints and derive link slopes across wheel travel."""

        if travel_limit_m <= 0.0 or not np.isfinite(travel_limit_m):
            raise ValueError("travel_limit_m must be finite and positive.")
        if sample_count < 5 or sample_count % 2 == 0:
            raise ValueError("sample_count must be an odd integer of at least five.")
        grid = np.linspace(-travel_limit_m, travel_limit_m, sample_count)
        return cls(
            front=_derive_axle_curve(vehicle, "front", grid),
            rear=_derive_axle_curve(vehicle, "rear", grid),
        )

    def instant_links_at(self, jounce_m: ArrayLike) -> DoubleWishboneInstantLinks:
        """Construct the four current instant links from corner jounce."""

        jounce = np.asarray(jounce_m, dtype=float)
        if jounce.shape != (4,) or not np.all(np.isfinite(jounce)):
            raise ValueError("Corner jounce must be a finite FL, FR, RL, RR four-vector.")
        front_left = self.front.at(float(jounce[0]))
        front_right = self.front.at(float(jounce[1]))
        rear_left = self.rear.at(float(jounce[2]))
        rear_right = self.rear.at(float(jounce[3]))
        return DoubleWishboneInstantLinks(
            corners=(
                CornerInstantLink(front_left[0], front_left[1]),
                CornerInstantLink(front_right[0], -front_right[1]),
                CornerInstantLink(rear_left[0], rear_left[1]),
                CornerInstantLink(rear_right[0], -rear_right[1]),
            )
        )


def project_double_wishbone_instant_links(
    vehicle: Mapping[str, Any],
    *,
    jounce_step_m: float = 1e-5,
) -> DoubleWishboneInstantLinks:
    """Linearize the active hardpoints into four instantaneous force links.

    ``CornerKinematics`` solves the same rigid upper-arm, lower-arm, upright,
    and tie-rod constraints represented by BobLib. A centered jounce derivative
    avoids the invalid side-view result that can arise by simply projecting
    swept wishbone pivot axes into 2D.
    """

    if not np.isfinite(jounce_step_m) or jounce_step_m <= 0.0:
        raise ValueError("jounce_step_m must be finite and positive.")
    geometry = DoubleWishboneGeometry.from_vehicle(
        vehicle,
        travel_limit_m=2.0 * jounce_step_m,
        sample_count=5,
    )
    return geometry.instant_links_at(np.zeros(4))


def _derive_axle_curve(
    vehicle: Mapping[str, Any],
    axle: str,
    jounce_grid: FloatArray,
) -> _AxleInstantLinkCurve:
    """Derive reciprocal force slopes from the constrained contact-patch tangent."""

    corner = CornerKinematics.from_vehicle(dict(vehicle), axle)
    contact_points: list[FloatArray] = []
    guess = np.zeros(3)
    for jounce in jounce_grid:
        guess, point_set, _residual = corner.solve_jounce(float(jounce), guess)
        contact_points.append(np.asarray(point_set.contact_patch, dtype=float))

    contact = np.asarray(contact_points, dtype=float)
    tangent = np.gradient(contact, jounce_grid, axis=0, edge_order=2)
    if not np.all(np.isfinite(tangent)) or np.any(np.abs(tangent[:, 2]) < 1e-9):
        raise ValueError(f"Could not derive finite {axle} instant-link tangents.")
    longitudinal = -tangent[:, 0] / tangent[:, 2]
    lateral = -tangent[:, 1] / tangent[:, 2]
    return _AxleInstantLinkCurve(
        jounce_m=tuple(float(value) for value in jounce_grid),
        longitudinal=tuple(float(value) for value in longitudinal),
        lateral_left=tuple(float(value) for value in lateral),
    )
