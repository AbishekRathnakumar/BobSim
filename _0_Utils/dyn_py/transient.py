"""Time integration and BobLib-compatible signal comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from _0_Utils.dyn_py.models import ModelInputs, ReducedVehicleModel


FloatArray = NDArray[np.float64]
ControlLaw = Callable[[float, FloatArray], ModelInputs]


@dataclass(frozen=True)
class TransientResult:
    time_s: FloatArray
    state: FloatArray
    state_names: tuple[str, ...]
    signals: Mapping[str, FloatArray]
    success: bool
    message: str


def simulate_transient(
    model: ReducedVehicleModel,
    *,
    initial_state: ArrayLike,
    controls: ModelInputs | ControlLaw,
    time_s: ArrayLike,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> TransientResult:
    """Integrate any model in the family with a constant or callable input."""

    evaluation_times = np.asarray(time_s, dtype=float)
    if evaluation_times.ndim != 1 or evaluation_times.size < 2:
        raise ValueError("time_s must contain at least two samples.")
    if np.any(np.diff(evaluation_times) <= 0.0):
        raise ValueError("time_s must be strictly increasing.")
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (model.state_size,):
        raise ValueError(
            f"Initial state must have shape ({model.state_size},), got {initial.shape}."
        )

    def input_at(time: float, state: FloatArray) -> ModelInputs:
        if isinstance(controls, ModelInputs):
            return controls
        return controls(time, state)

    def rhs(time: float, state: FloatArray) -> FloatArray:
        return model.derivative(time, state, input_at(time, state))

    solution = solve_ivp(  # type: ignore[call-overload]
        rhs,
        (float(evaluation_times[0]), float(evaluation_times[-1])),
        initial,
        t_eval=evaluation_times,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    states = np.asarray(solution.y.T, dtype=float)
    signals: dict[str, FloatArray] = {
        name: states[:, index] for index, name in enumerate(model.state_names)
    }
    body_acceleration = np.empty((len(solution.t), 3), dtype=float)
    steering = np.empty(len(solution.t), dtype=float)
    camber = np.empty((len(solution.t), 4), dtype=float)
    toe = np.empty((len(solution.t), 4), dtype=float)
    contact_patch = np.empty((len(solution.t), 4, 3), dtype=float)
    for index, (time, state) in enumerate(zip(solution.t, states)):
        model_inputs = input_at(float(time), state)
        output = model.evaluate(state, model_inputs)
        velocity = state[model.dof :]
        if model.dof == 3:
            u, v, yaw_rate = velocity
        else:
            u, v, yaw_rate = velocity[0], velocity[1], velocity[5]
        body_acceleration[index] = (
            output.generalized_acceleration[0] - yaw_rate * v,
            output.generalized_acceleration[1] + yaw_rate * u,
            output.generalized_acceleration[2] if model.dof == 3 else output.generalized_acceleration[5],
        )
        steering[index] = model_inputs.steering_rad
        camber[index] = output.camber_rad
        toe[index] = output.toe_rad
        contact_patch[index] = output.contact_patch_positions_body_m
    signals.update(
        {
            "accX": body_acceleration[:, 0],
            "accY": body_acceleration[:, 1],
            "yawAccel": body_acceleration[:, 2],
            "handwheelAngle": steering,
            "velX": signals["u"],
            "velY": signals["v"],
            "yawVel": signals["yaw_rate"],
            "sideslip": np.arctan2(signals["v"], np.maximum(np.abs(signals["u"]), 1e-9)),
            "roll": signals.get("roll", np.zeros(len(solution.t))),
        }
    )
    for corner_index, corner in enumerate(("FL", "FR", "RL", "RR")):
        signals[f"camber{corner}"] = camber[:, corner_index]
        signals[f"toe{corner}"] = toe[:, corner_index]
        signals[f"contactPatchX{corner}"] = contact_patch[:, corner_index, 0]
        signals[f"contactPatchY{corner}"] = contact_patch[:, corner_index, 1]
        signals[f"contactPatchZ{corner}"] = contact_patch[:, corner_index, 2]
    return TransientResult(
        time_s=np.asarray(solution.t, dtype=float),
        state=states,
        state_names=model.state_names,
        signals=signals,
        success=bool(solution.success),
        message=str(solution.message),
    )


def compare_transient_signals(
    reference_time_s: ArrayLike,
    reference_signals: Mapping[str, ArrayLike],
    candidate: TransientResult,
) -> dict[str, dict[str, float]]:
    """Compare reduced-model histories with BobLib result CSV channels."""

    reference_time = np.asarray(reference_time_s, dtype=float)
    if reference_time.ndim != 1 or reference_time.size < 2:
        raise ValueError("Reference time must contain at least two samples.")
    metrics: dict[str, dict[str, float]] = {}
    for name, raw_reference in reference_signals.items():
        if name not in candidate.signals:
            continue
        reference = np.asarray(raw_reference, dtype=float)
        if reference.shape != reference_time.shape:
            raise ValueError(f"Reference signal {name!r} does not match reference time.")
        predicted = np.interp(reference_time, candidate.time_s, candidate.signals[name])
        error = predicted - reference
        span = float(np.ptp(reference))
        rms = float(np.sqrt(np.mean(error**2)))
        metrics[name] = {
            "rmse": rms,
            "normalized_rmse": rms / max(span, 1e-12),
            "max_abs_error": float(np.max(np.abs(error))),
            "bias": float(np.mean(error)),
        }
    return metrics
