"""Discrete Mori diagnostics for bias-centered relative cage coordinates."""

from __future__ import annotations

import numpy as np


def bias_centered_phase_state(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    bias: np.ndarray,
) -> np.ndarray:
    """Return ``[u-u0, p]`` with particle components as ensemble samples."""

    position = np.asarray(relative_position, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    cage_bias = np.asarray(bias, dtype=float)
    if (
        position.ndim != 3
        or position.shape[-1] != 3
        or velocity.shape != position.shape
        or cage_bias.shape != position.shape[1:]
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(cage_bias))
    ):
        raise ValueError("relative paths and bias must be finite and aligned")
    centered = (position - cage_bias[None, :, :]).reshape(len(position), -1)
    momentum = velocity.reshape(len(velocity), -1)
    return np.stack([centered, momentum], axis=2)


def discrete_mori_noise(
    resolved_state: np.ndarray,
    operators: np.ndarray,
) -> np.ndarray:
    """Reconstruct time-origin-conditioned discrete Mori noise ``W_k``.

    For every shared time origin ``i`` and order ``k``, this evaluates

    ``W_k|i = g_(i+k+1) - sum_(ell=0)^k Omega_ell g_(i+k-ell)``.

    All orders use the same origins so their cross-correlations are aligned.
    """

    state = np.asarray(resolved_state, dtype=float)
    memory = np.asarray(operators, dtype=float)
    if (
        state.ndim != 3
        or memory.ndim != 3
        or memory.shape[1] != memory.shape[2]
        or state.shape[2] != memory.shape[1]
        or len(memory) < 1
        or len(state) < len(memory) + 1
        or np.any(~np.isfinite(state))
        or np.any(~np.isfinite(memory))
    ):
        raise ValueError("resolved state and square operators must be finite and aligned")
    origin_count = len(state) - len(memory)
    noise = np.empty((len(memory), origin_count, state.shape[1], state.shape[2]))
    origins = np.arange(origin_count)
    for order in range(len(memory)):
        predicted = np.zeros((origin_count, state.shape[1], state.shape[2]))
        for lag in range(order + 1):
            past = state[origins + order - lag]
            predicted += past @ memory[lag].T
        noise[order] = state[origins + order + 1] - predicted
    return noise


def discrete_mori_gfd_diagnostic(
    resolved_state: np.ndarray,
    operators: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Test Mori orthogonality and the discrete generalized FDT on held data.

    For positive order ``k``, the discrete generalized fluctuation-dissipation
    relation is

    ``Omega_k = -<W_k W_0^T> C(-Delta)^-1``.

    This is stronger than correlation propagation: it couples the inferred
    memory operators to the time-origin-conditioned orthogonal dynamics.
    """

    state = np.asarray(resolved_state, dtype=float)
    memory = np.asarray(operators, dtype=float)
    if len(memory) < 2:
        raise ValueError("at least one positive-order memory operator is required")
    noise = discrete_mori_noise(state, memory)
    origin_count = noise.shape[1]
    initial = state[:origin_count]
    sample_count = state.shape[1]
    initial_covariance = np.einsum("tni,tnj->ij", initial, initial) / (
        origin_count * sample_count
    )
    initial_scale = np.sqrt(np.maximum(np.diag(initial_covariance), 1e-30))
    maximum_orthogonality = 0.0
    noise_initial_correlation = np.empty(
        (len(memory), state.shape[2], state.shape[2]), dtype=float
    )
    for order in range(len(memory)):
        covariance = np.einsum("tni,tnj->ij", noise[order], noise[order]) / (
            origin_count * sample_count
        )
        noise_scale = np.sqrt(np.maximum(np.diag(covariance), 1e-30))
        cross = np.einsum("tni,tnj->ij", noise[order], initial) / (
            origin_count * sample_count
        )
        correlation = cross / np.outer(noise_scale, initial_scale)
        noise_initial_correlation[order] = correlation
        maximum_orthogonality = max(
            maximum_orthogonality, float(np.max(np.abs(correlation)))
        )

    lag_minus = np.einsum("tni,tnj->ij", state[:-1], state[1:]) / (
        (len(state) - 1) * sample_count
    )
    if np.linalg.matrix_rank(lag_minus) < state.shape[2]:
        raise ValueError("negative-lag correlation is singular")
    inverse_lag_minus = np.linalg.inv(lag_minus)
    gfd_operators = np.empty_like(memory[1:])
    for order in range(1, len(memory)):
        force_correlation = np.einsum(
            "tni,tnj->ij", noise[order], noise[0]
        ) / (origin_count * sample_count)
        gfd_operators[order - 1] = -force_correlation @ inverse_lag_minus
    target = memory[1:]
    difference = gfd_operators - target
    normalized_rmse = float(
        np.linalg.norm(difference) / max(np.linalg.norm(target), 1e-30)
    )
    flat_target = target.ravel()
    flat_gfd = gfd_operators.ravel()
    shape_correlation = (
        float(np.corrcoef(flat_target, flat_gfd)[0, 1])
        if np.std(flat_target) > 0.0 and np.std(flat_gfd) > 0.0
        else float("nan")
    )
    return {
        "noise": noise,
        "noise_initial_state_correlation": noise_initial_correlation,
        "gfd_operators": gfd_operators,
        "maximum_noise_initial_state_correlation": maximum_orthogonality,
        "gfd_operator_normalized_rmse": normalized_rmse,
        "gfd_operator_shape_correlation": shape_correlation,
        "gfd_operator_maximum_absolute_error": float(np.max(np.abs(difference))),
        "lag_minus_correlation_condition_number": float(np.linalg.cond(lag_minus)),
        "thermodynamic_claim_allowed": 0.0,
    }


def propagate_discrete_mori_correlation(
    operators: np.ndarray,
    *,
    initial_correlation: np.ndarray,
    output_count: int,
) -> np.ndarray:
    """Continue a matrix correlation sequence with fixed finite MZ memory."""

    memory = np.asarray(operators, dtype=float)
    initial = np.asarray(initial_correlation, dtype=float)
    if (
        memory.ndim != 3
        or memory.shape[1] != memory.shape[2]
        or initial.ndim != 3
        or initial.shape[1:] != memory.shape[1:]
        or len(initial) < len(memory) + 1
        or output_count < len(initial)
        or np.any(~np.isfinite(memory))
        or np.any(~np.isfinite(initial))
    ):
        raise ValueError("operators, initial correlations, and output count must align")
    output = np.empty((output_count, *initial.shape[1:]))
    output[: len(initial)] = initial
    for index in range(len(initial), output_count):
        output[index] = sum(
            memory[lag] @ output[index - 1 - lag]
            for lag in range(len(memory))
        )
    return output
