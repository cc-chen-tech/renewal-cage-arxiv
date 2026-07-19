#!/usr/bin/env python3
"""Run the frozen nonlinear auxiliary-bath experiment on remote compute only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nonlinear_bath_gle import (  # noqa: E402
    NonlinearBathControls,
    nonlinear_bath_step,
    periodic_potential,
)


MODES = (
    "canary",
    "canary-half-step",
    "production",
    "null-constant-coupling",
    "null-no-bath",
)
_CLOSED_CLAIMS = {
    "exact_nonlinear_bath_elimination_supported": 0.0,
    "synthetic_bath_level_fdt_replay_supported": 0.0,
    "synthetic_delayed_hazard_emerges": 0.0,
    "real_ka_position_dependent_kernel_authorized": 0.0,
    "autonomous_single_particle_gle_allowed": 0.0,
    "complete_event_clock_closure_allowed": 0.0,
    "kramers_escape_claim_allowed": 0.0,
    "spatial_facilitation_claim_allowed": 0.0,
    "thermodynamic_claim_allowed": 0.0,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_simulation_protocol(mode: str) -> dict[str, object]:
    """Return the immutable controls for one primary or null execution mode."""

    if mode not in MODES:
        raise ValueError("unknown nonlinear-bath simulation mode")
    time_step = 5e-4 if mode == "canary-half-step" else 1e-3
    is_canary = mode in {"canary", "canary-half-step"}
    modulation = np.array([0.45, 0.25])
    amplitudes = np.array([1.00, 0.55])
    if mode == "null-constant-coupling":
        modulation = np.zeros(2)
    if mode == "null-no-bath":
        amplitudes = np.zeros(2)
    controls = NonlinearBathControls(
        temperature=0.58,
        friction=1.0,
        period=1.0,
        barrier=1.74,
        rates=np.array([0.20, 1.00]),
        amplitudes=amplitudes,
        modulation=modulation,
        phases=np.array([0.0, 0.5 * np.pi]),
        time_step=time_step,
    )
    if is_canary:
        burn_in_steps = 0
        production_steps = int(round(2.0 / time_step))
        trajectory_count = 16
        event_stride = 1
        equilibrium_stride = 1
    else:
        burn_in_steps = int(round(100.0 / time_step))
        production_steps = int(round(400.0 / time_step))
        trajectory_count = 256
        event_stride = int(round(0.01 / time_step))
        equilibrium_stride = int(round(0.10 / time_step))
    return {
        "mode": mode,
        "controls": controls,
        "seed": 20260811,
        "trajectory_count": trajectory_count,
        "burn_in_steps": burn_in_steps,
        "production_steps": production_steps,
        "requested_step_count": burn_in_steps + production_steps,
        "event_sample_stride": event_stride,
        "equilibrium_sample_stride": equilibrium_stride,
        "event_sample_time": event_stride * time_step,
        "equilibrium_sample_time": equilibrium_stride * time_step,
        "potential_amplitude": float(controls.barrier),
        "physical_barrier_height": float(2.0 * controls.barrier),
    }


def checkpoint_metadata(
    *,
    frozen_simulation_protocol: str,
    source_sha256: str,
    requested_step_count: int,
) -> dict[str, object]:
    """Return every field that must match before a checkpoint can resume."""

    protocol = globals()["frozen_simulation_protocol"](
        frozen_simulation_protocol
    )
    controls = protocol["controls"]
    if not isinstance(controls, NonlinearBathControls):
        raise ValueError("invalid frozen controls")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) < 1
        or isinstance(requested_step_count, bool)
        or not isinstance(requested_step_count, (int, np.integer))
        or requested_step_count < 1
        or requested_step_count != int(protocol["requested_step_count"])
    ):
        raise ValueError("checkpoint metadata inputs are invalid")
    return {
        "frozen_simulation_protocol": str(frozen_simulation_protocol),
        "source_sha256": source_sha256,
        "gle_source_sha256": file_sha256(ROOT / "src" / "nonlinear_bath_gle.py"),
        "requested_step_count": float(requested_step_count),
        "seed": float(protocol["seed"]),
        "trajectory_count": float(protocol["trajectory_count"]),
        "burn_in_steps": float(protocol["burn_in_steps"]),
        "production_steps": float(protocol["production_steps"]),
        "event_sample_stride": float(protocol["event_sample_stride"]),
        "equilibrium_sample_stride": float(protocol["equilibrium_sample_stride"]),
        "temperature": float(controls.temperature),
        "friction": float(controls.friction),
        "period": float(controls.period),
        "barrier": float(controls.barrier),
        "potential_amplitude": float(protocol["potential_amplitude"]),
        "physical_barrier_height": float(protocol["physical_barrier_height"]),
        "rates": controls.rates.copy(),
        "amplitudes": controls.amplitudes.copy(),
        "modulation": controls.modulation.copy(),
        "phases": controls.phases.copy(),
        "time_step": float(controls.time_step),
    }


def validate_checkpoint_metadata(
    saved: dict[str, object],
    expected: dict[str, object],
) -> None:
    """Reject any changed source, mode, seed, size, or physical control."""

    for key, expected_value in expected.items():
        if key not in saved:
            raise ValueError(f"checkpoint provenance is missing {key}")
        actual = np.asarray(saved[key])
        target = np.asarray(expected_value)
        if actual.shape != target.shape or not np.array_equal(actual, target):
            raise ValueError(f"checkpoint provenance mismatch: {key}")


def require_remote_execution() -> None:
    """Prevent accidental local production or canary trajectory generation."""

    if os.environ.get("RENEWAL_CAGE_REMOTE_COMPUTE") != "1":
        raise RuntimeError(
            "nonlinear-bath simulation is remote-only; set the remote compute marker"
        )


def _sample_equilibrium_positions(
    rng: np.random.Generator,
    count: int,
    controls: NonlinearBathControls,
) -> np.ndarray:
    grid = np.linspace(-0.5 * controls.period, 0.5 * controls.period, 65537)
    weight = np.exp(
        -periodic_potential(
            grid,
            barrier=controls.barrier,
            period=controls.period,
        )
        / controls.temperature
    )
    cdf = np.cumsum(weight)
    cdf /= cdf[-1]
    return np.interp(rng.random(count), cdf, grid)


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _checkpoint_integer(
    saved: Mapping[str, object],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if key not in saved:
        raise ValueError(f"checkpoint payload is missing {key}")
    value = np.asarray(saved[key])
    if value.shape != ():
        raise ValueError(f"checkpoint payload has non-scalar {key}")
    number = float(value)
    if (
        not math.isfinite(number)
        or number != round(number)
        or number < minimum
        or number > maximum
    ):
        raise ValueError(f"checkpoint payload has invalid {key}")
    return int(number)


def _checkpoint_array(
    saved: Mapping[str, object],
    key: str,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    if key not in saved:
        raise ValueError(f"checkpoint payload is missing {key}")
    value = np.asarray(saved[key], dtype=float)
    if value.shape != shape or np.any(~np.isfinite(value)):
        raise ValueError(f"checkpoint payload has invalid {key}")
    return value.copy()


def validate_checkpoint_payload(
    saved: Mapping[str, object],
    *,
    provenance: dict[str, object],
    protocol: dict[str, object],
) -> dict[str, object]:
    """Validate all resume counts, arrays, RNG state, and closed claims."""

    validate_checkpoint_metadata(
        {key: saved[key] for key in provenance if key in saved},
        provenance,
    )
    controls = protocol.get("controls")
    if not isinstance(controls, NonlinearBathControls):
        raise ValueError("checkpoint payload has invalid frozen controls")
    requested = int(protocol["requested_step_count"])
    trajectories = int(protocol["trajectory_count"])
    burn_in = int(protocol["burn_in_steps"])
    event_stride = int(protocol["event_sample_stride"])
    equilibrium_stride = int(protocol["equilibrium_sample_stride"])
    mode_count = len(controls.rates)
    completed = _checkpoint_integer(
        saved,
        "completed_step_count",
        minimum=0,
        maximum=requested,
    )
    event_count = _checkpoint_integer(
        saved,
        "stored_event_sample_count",
        minimum=0,
        maximum=int(protocol["production_steps"]) // event_stride + 1,
    )
    equilibrium_count = _checkpoint_integer(
        saved,
        "stored_equilibrium_sample_count",
        minimum=0,
        maximum=int(protocol["production_steps"]) // equilibrium_stride + 1,
    )

    def expected_count(stride: int) -> int:
        if completed < burn_in:
            return 0
        return (completed - burn_in) // stride + 1

    if event_count != expected_count(event_stride):
        raise ValueError("checkpoint payload has inconsistent event sample count")
    if equilibrium_count != expected_count(equilibrium_stride):
        raise ValueError(
            "checkpoint payload has inconsistent equilibrium sample count"
        )
    cache_complete = _checkpoint_integer(
        saved,
        "cache_complete",
        minimum=0,
        maximum=1,
    )
    if cache_complete != int(completed == requested):
        raise ValueError("checkpoint payload has inconsistent completion flag")

    arrays = {
        "current_position": _checkpoint_array(
            saved,
            "current_position",
            shape=(trajectories,),
        ),
        "current_momentum": _checkpoint_array(
            saved,
            "current_momentum",
            shape=(trajectories,),
        ),
        "current_auxiliary": _checkpoint_array(
            saved,
            "current_auxiliary",
            shape=(trajectories, mode_count),
        ),
        "event_positions": _checkpoint_array(
            saved,
            "event_positions",
            shape=(event_count, trajectories),
        ),
        "equilibrium_positions": _checkpoint_array(
            saved,
            "equilibrium_positions",
            shape=(equilibrium_count, trajectories),
        ),
        "equilibrium_momenta": _checkpoint_array(
            saved,
            "equilibrium_momenta",
            shape=(equilibrium_count, trajectories),
        ),
        "equilibrium_auxiliary": _checkpoint_array(
            saved,
            "equilibrium_auxiliary",
            shape=(equilibrium_count, trajectories, mode_count),
        ),
    }
    is_canary = str(protocol["mode"]) in {"canary", "canary-half-step"}
    arrays["canary_normal_p"] = _checkpoint_array(
        saved,
        "canary_normal_p",
        shape=(completed if is_canary else 0, trajectories),
    )
    arrays["canary_normal_z"] = _checkpoint_array(
        saved,
        "canary_normal_z",
        shape=(completed if is_canary else 0, trajectories, mode_count),
    )
    for key in _CLOSED_CLAIMS:
        if key not in saved or np.asarray(saved[key]).shape != ():
            raise ValueError(f"checkpoint payload is missing closed claim {key}")
        if float(np.asarray(saved[key])) != 0.0:
            raise ValueError(f"checkpoint payload has open broad claim {key}")

    if "rng_state_json" not in saved:
        raise ValueError("checkpoint payload is missing RNG state")
    encoded_rng = np.asarray(saved["rng_state_json"])
    if encoded_rng.shape != () or not isinstance(encoded_rng.item(), str):
        raise ValueError("checkpoint payload has invalid RNG state")
    try:
        rng_state = json.loads(encoded_rng.item())
        probe = np.random.default_rng()
        probe.bit_generator.state = rng_state
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("checkpoint payload has invalid RNG state") from error
    return {
        "completed_step_count": completed,
        "event_sample_count": event_count,
        "equilibrium_sample_count": equilibrium_count,
        "rng_state": rng_state,
        **arrays,
    }


def run_simulation(
    output_path: Path,
    *,
    mode: str,
    checkpoint_interval: int,
    resume: bool,
) -> dict[str, object]:
    """Run or resume one frozen vectorized remote trajectory ensemble."""

    require_remote_execution()
    if (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, (int, np.integer))
        or checkpoint_interval < 1
    ):
        raise ValueError("checkpoint interval must be a positive integer")
    protocol = frozen_simulation_protocol(mode)
    controls = protocol["controls"]
    if not isinstance(controls, NonlinearBathControls):
        raise ValueError("frozen protocol controls are invalid")
    source_hash = file_sha256(Path(__file__))
    requested = int(protocol["requested_step_count"])
    provenance = checkpoint_metadata(
        frozen_simulation_protocol=mode,
        source_sha256=source_hash,
        requested_step_count=requested,
    )
    trajectory_count = int(protocol["trajectory_count"])
    burn_in = int(protocol["burn_in_steps"])
    production_steps = int(protocol["production_steps"])
    event_stride = int(protocol["event_sample_stride"])
    equilibrium_stride = int(protocol["equilibrium_sample_stride"])
    event_capacity = production_steps // event_stride + 1
    equilibrium_capacity = production_steps // equilibrium_stride + 1
    is_canary = mode in {"canary", "canary-half-step"}

    event_positions = np.empty((event_capacity, trajectory_count), dtype=float)
    equilibrium_positions = np.empty(
        (equilibrium_capacity, trajectory_count), dtype=float
    )
    equilibrium_momenta = np.empty_like(equilibrium_positions)
    equilibrium_auxiliary = np.empty(
        (equilibrium_capacity, trajectory_count, len(controls.rates)),
        dtype=float,
    )
    canary_normal_p = (
        np.empty((requested, trajectory_count), dtype=float)
        if is_canary
        else np.empty((0, trajectory_count), dtype=float)
    )
    canary_normal_z = (
        np.empty((requested, trajectory_count, len(controls.rates)), dtype=float)
        if is_canary
        else np.empty((0, trajectory_count, len(controls.rates)), dtype=float)
    )

    if output_path.exists():
        if not resume:
            raise ValueError("output cache exists; --resume is required")
        with np.load(output_path, allow_pickle=False) as cache:
            payload = validate_checkpoint_payload(
                cache,
                provenance=provenance,
                protocol=protocol,
            )
            completed = int(payload["completed_step_count"])
            event_count = int(payload["event_sample_count"])
            equilibrium_count = int(payload["equilibrium_sample_count"])
            position = np.asarray(payload["current_position"])
            momentum = np.asarray(payload["current_momentum"])
            auxiliary = np.asarray(payload["current_auxiliary"])
            rng = np.random.default_rng()
            rng.bit_generator.state = payload["rng_state"]
            event_positions[:event_count] = payload["event_positions"]
            equilibrium_positions[:equilibrium_count] = payload[
                "equilibrium_positions"
            ]
            equilibrium_momenta[:equilibrium_count] = payload[
                "equilibrium_momenta"
            ]
            equilibrium_auxiliary[:equilibrium_count] = payload[
                "equilibrium_auxiliary"
            ]
            if is_canary:
                canary_normal_p[:completed] = payload["canary_normal_p"]
                canary_normal_z[:completed] = payload["canary_normal_z"]
    else:
        if resume:
            raise ValueError("--resume requires an existing output cache")
        completed = 0
        event_count = 0
        equilibrium_count = 0
        rng = np.random.default_rng(int(protocol["seed"]))
        position = _sample_equilibrium_positions(rng, trajectory_count, controls)
        momentum = rng.normal(
            scale=math.sqrt(controls.temperature),
            size=trajectory_count,
        )
        auxiliary = rng.normal(
            scale=math.sqrt(controls.temperature),
            size=(trajectory_count, len(controls.rates)),
        )

    def retain_sample(step_count: int) -> None:
        nonlocal event_count, equilibrium_count
        if step_count < burn_in:
            return
        production_index = step_count - burn_in
        if production_index % event_stride == 0:
            event_positions[event_count] = position
            event_count += 1
        if production_index % equilibrium_stride == 0:
            equilibrium_positions[equilibrium_count] = position
            equilibrium_momenta[equilibrium_count] = momentum
            equilibrium_auxiliary[equilibrium_count] = auxiliary
            equilibrium_count += 1

    if completed == 0:
        retain_sample(0)

    def save() -> None:
        _atomic_savez(
            output_path,
            **provenance,
            completed_step_count=float(completed),
            stored_event_sample_count=float(event_count),
            stored_equilibrium_sample_count=float(equilibrium_count),
            cache_complete=float(completed == requested),
            current_position=position,
            current_momentum=momentum,
            current_auxiliary=auxiliary,
            event_positions=event_positions[:event_count],
            equilibrium_positions=equilibrium_positions[:equilibrium_count],
            equilibrium_momenta=equilibrium_momenta[:equilibrium_count],
            equilibrium_auxiliary=equilibrium_auxiliary[:equilibrium_count],
            canary_normal_p=canary_normal_p[:completed] if is_canary else canary_normal_p,
            canary_normal_z=canary_normal_z[:completed] if is_canary else canary_normal_z,
            rng_state_json=np.asarray(json.dumps(rng.bit_generator.state)),
            **_CLOSED_CLAIMS,
        )

    for step_index in range(completed, requested):
        normal_p = rng.normal(size=trajectory_count)
        normal_z = rng.normal(size=(trajectory_count, len(controls.rates)))
        if is_canary:
            canary_normal_p[step_index] = normal_p
            canary_normal_z[step_index] = normal_z
        result = nonlinear_bath_step(
            position,
            momentum,
            auxiliary,
            normal_p=normal_p,
            normal_z=normal_z,
            controls=controls,
        )
        position = np.asarray(result["position"])
        momentum = np.asarray(result["momentum"])
        auxiliary = np.asarray(result["auxiliary"])
        completed = step_index + 1
        retain_sample(completed)
        if completed % checkpoint_interval == 0 or completed == requested:
            save()
            print(f"{output_path.name}: step {completed}/{requested}", flush=True)
    return {
        "cache": str(output_path.resolve()),
        "mode": mode,
        "completed_step_count": float(completed),
        "requested_step_count": float(requested),
        "cache_complete": float(completed == requested),
        **_CLOSED_CLAIMS,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=50000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint interval must be positive")
    require_remote_execution()
    result = run_simulation(
        args.output_path,
        mode=args.mode,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    print(result)


if __name__ == "__main__":
    main()
