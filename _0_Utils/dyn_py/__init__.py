"""``dyn_py`` reduced-order dynamics shared by transient and envelope workflows.

The model names count generalized coordinates, not first-order states:

* 3DOF: global x/y/yaw planar body motion.
* 6DOF: full rigid-body translation and rotation.
* 10DOF: 6DOF body plus four wheel rotations.
* 14DOF: 10DOF plus four unsprung vertical motions.
"""

from _0_Utils.dyn_py.models import (
    DOFModel,
    ModelInputs,
    ModelOutput,
    VehicleDynamicsSystem,
    VehicleModel3DOF,
    VehicleModel6DOF,
    VehicleModel10DOF,
    VehicleModel14DOF,
    create_model,
)
from _0_Utils.dyn_py.parameters import (
    CORNERS,
    PowertrainLimits,
    ReducedVehicleOverrides,
    ReducedVehicleParameters,
    TireParameters,
    apply_reduced_vehicle_overrides,
    load_reduced_vehicle_parameters,
    project_powertrain_limits,
)
from _0_Utils.dyn_py.qss import (
    QSSResult,
    solve_acceleration_trim,
    solve_moment_state,
    steady_state_residual,
    solve_steady_state,
)
from _0_Utils.dyn_py.suspension import (
    CornerInstantLink,
    DoubleWishboneGeometry,
    DoubleWishboneInstantLinks,
    project_double_wishbone_instant_links,
)
from _0_Utils.dyn_py.transient import (
    TransientResult,
    compare_transient_signals,
    simulate_transient,
)

__all__ = [
    "CORNERS",
    "CornerInstantLink",
    "DOFModel",
    "DoubleWishboneGeometry",
    "DoubleWishboneInstantLinks",
    "ModelInputs",
    "ModelOutput",
    "PowertrainLimits",
    "QSSResult",
    "ReducedVehicleOverrides",
    "ReducedVehicleParameters",
    "TireParameters",
    "TransientResult",
    "VehicleDynamicsSystem",
    "VehicleModel3DOF",
    "VehicleModel6DOF",
    "VehicleModel10DOF",
    "VehicleModel14DOF",
    "apply_reduced_vehicle_overrides",
    "compare_transient_signals",
    "create_model",
    "load_reduced_vehicle_parameters",
    "project_double_wishbone_instant_links",
    "project_powertrain_limits",
    "simulate_transient",
    "solve_acceleration_trim",
    "solve_moment_state",
    "solve_steady_state",
    "steady_state_residual",
]
