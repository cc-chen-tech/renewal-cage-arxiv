"""Collective velocity embeddings for a tagged-particle center GLE."""

from __future__ import annotations

import math

import numpy as np


def _ngp_3d(vectors: np.ndarray) -> float:
    squared = np.sum(vectors**2, axis=1)
    second = float(np.mean(squared))
    return 3.0 * float(np.mean(squared**2)) / (5.0 * second**2) - 1.0 if second > 0.0 else 0.0


def nearest_outer_bath_state(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    target_index: int,
    active_indices: np.ndarray,
    box_lengths: np.ndarray,
    neighbor_count: int,
) -> dict[str, np.ndarray]:
    """Resolve the nearest non-active bath particles around one tagged particle.

    The active IDs are fixed by a reference configuration.  At each supplied
    frame, this returns the ``neighbor_count`` closest remaining particle IDs,
    with minimum-image relative positions and instantaneous relative
    velocities.  The rank order can change in time; it is deliberately a
    resolved spatial bath state rather than a persistent particle label.
    """

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    active = np.asarray(active_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or not len(positions) or not np.all(np.isfinite(positions)):
        raise ValueError("positions must be a finite (frames, particles, 3) array")
    if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
        raise ValueError("velocities must be finite and align with positions")
    if target_index < 0 or target_index >= positions.shape[1]:
        raise ValueError("target_index must select a valid particle")
    if active.ndim != 1 or not len(active) or np.any(active < 0) or np.any(active >= positions.shape[1]):
        raise ValueError("active_indices must select valid particles")
    if len(np.unique(active)) != len(active) or target_index not in active:
        raise ValueError("active_indices must be unique and contain target_index")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be positive")
    outer = np.setdiff1d(np.arange(positions.shape[1], dtype=int), active, assume_unique=False)
    if neighbor_count < 1 or neighbor_count > len(outer):
        raise ValueError("neighbor_count must select one or more non-active particles")

    relative_position = positions[:, outer] - positions[:, target_index : target_index + 1]
    relative_position -= box_lengths * np.rint(relative_position / box_lengths)
    relative_velocity = velocities[:, outer] - velocities[:, target_index : target_index + 1]
    squared_distance = np.sum(relative_position**2, axis=2)
    candidate = np.argpartition(squared_distance, neighbor_count - 1, axis=1)[:, :neighbor_count]
    candidate_distance = np.take_along_axis(squared_distance, candidate, axis=1)
    rank = np.argsort(candidate_distance, axis=1)
    selection = np.take_along_axis(candidate, rank, axis=1)
    return {
        "particle_indices": outer[selection],
        "relative_positions": np.take_along_axis(relative_position, selection[..., None], axis=1),
        "relative_velocities": np.take_along_axis(relative_velocity, selection[..., None], axis=1),
    }


def fixed_bath_particle_state(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    target_index: int,
    particle_indices: np.ndarray,
    box_lengths: np.ndarray,
) -> dict[str, np.ndarray]:
    """Resolve a fixed labelled bath subset relative to a tagged particle."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    particles = np.asarray(particle_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or not len(positions) or not np.all(np.isfinite(positions)):
        raise ValueError("positions must be a finite (frames, particles, 3) array")
    if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
        raise ValueError("velocities must be finite and align with positions")
    if target_index < 0 or target_index >= positions.shape[1]:
        raise ValueError("target_index must select a valid particle")
    if particles.ndim != 1 or not len(particles) or np.any(particles < 0) or np.any(particles >= positions.shape[1]):
        raise ValueError("particle_indices must select valid particles")
    if len(np.unique(particles)) != len(particles) or target_index in particles:
        raise ValueError("particle_indices must be unique and exclude target_index")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be positive")
    relative_position = positions[:, particles] - positions[:, target_index : target_index + 1]
    relative_position -= box_lengths * np.rint(relative_position / box_lengths)
    return {
        "particle_indices": particles,
        "relative_positions": relative_position,
        "relative_velocities": velocities[:, particles] - velocities[:, target_index : target_index + 1],
    }


def local_neighbor_velocity_field(
    positions: np.ndarray,
    *,
    target_indices: np.ndarray,
    box_lengths: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean current-interval velocity of a tagged particle's geometric neighbors."""

    positions = np.asarray(positions, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (at least 2 frames, particles, 3)")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= positions.shape[1]):
        raise ValueError("target_indices must be valid particle indices")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0) or cutoff <= 0.0:
        raise ValueError("box_lengths and cutoff must be positive")
    velocity = positions[1:] - positions[:-1]
    mean = np.zeros((len(velocity), len(target), 3), dtype=float)
    coordination = np.zeros((len(velocity), len(target)), dtype=int)
    for frame_index, frame in enumerate(positions[:-1]):
        wrapped = np.mod(frame, box_lengths)
        displacement = wrapped[target, None, :] - wrapped[None, :, :]
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        distance = np.sqrt(np.sum(displacement**2, axis=2))
        neighbor = (distance > 1e-10) & (distance < cutoff)
        for local_index in range(len(target)):
            selected = neighbor[local_index]
            coordination[frame_index, local_index] = int(np.sum(selected))
            if np.any(selected):
                mean[frame_index, local_index] = np.mean(velocity[frame_index, selected], axis=0)
    return mean, coordination


def local_neighbor_particle_velocity_field(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    target_indices: np.ndarray,
    box_lengths: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the instantaneous velocity field of a target's geometric neighbors.

    Unlike :func:`local_neighbor_velocity_field`, this uses the actual
    underdamped particle velocities recorded by the many-particle Langevin
    simulation.  It is therefore a resolved physical variable for a
    phase-space Mori--Zwanzig projection, not a finite-difference surrogate.
    """

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 1:
        raise ValueError("positions must have shape (at least 1 frame, particles, 3)")
    if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
        raise ValueError("velocities must be finite and align with positions")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= positions.shape[1]):
        raise ValueError("target_indices must be valid particle indices")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0) or cutoff <= 0.0:
        raise ValueError("box_lengths and cutoff must be positive")
    mean = np.zeros((len(positions), len(target), 3), dtype=float)
    coordination = np.zeros((len(positions), len(target)), dtype=int)
    for frame_index, frame in enumerate(positions):
        wrapped = np.mod(frame, box_lengths)
        displacement = wrapped[target, None, :] - wrapped[None, :, :]
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        distance = np.sqrt(np.sum(displacement**2, axis=2))
        neighbor = (distance > 1e-10) & (distance < cutoff)
        for target_slot, selected in enumerate(neighbor):
            coordination[frame_index, target_slot] = int(np.sum(selected))
            if np.any(selected):
                mean[frame_index, target_slot] = np.mean(velocities[frame_index, selected], axis=0)
    return mean, coordination


def local_affine_nonaffine_state(
    positions: np.ndarray,
    *,
    target_indices: np.ndarray,
    box_lengths: np.ndarray,
    cutoff: float,
) -> dict[str, np.ndarray]:
    """Measure the Falk-Langer local affine gradient and nonaffine residual.

    For each completed interval, relative neighbor displacements are fitted to
    ``Delta r_j-Delta r_i = L_i r_ij``.  The tensor ``L_i`` retains angular
    local deformation; ``D2min`` is the mean squared residual.  Both are
    causal history variables at the end of the interval.
    """

    positions = np.asarray(positions, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (at least 2 frames, particles, 3)")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= positions.shape[1]):
        raise ValueError("target_indices must select valid particles")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0) or cutoff <= 0.0:
        raise ValueError("box_lengths and cutoff must be positive")
    affine = np.full((len(positions) - 1, len(target), 3, 3), np.nan, dtype=float)
    d2min = np.full(affine.shape[:2], np.nan, dtype=float)
    valid = np.zeros(affine.shape[:2], dtype=bool)
    increment = positions[1:] - positions[:-1]
    for frame_index, frame in enumerate(positions[:-1]):
        wrapped = np.mod(frame, box_lengths)
        relative = wrapped[target, None, :] - wrapped[None, :, :]
        relative -= box_lengths * np.rint(relative / box_lengths)
        distance = np.linalg.norm(relative, axis=2)
        for local_index, particle in enumerate(target):
            neighbor = (distance[local_index] > 1e-10) & (distance[local_index] < cutoff)
            if np.sum(neighbor) < 3:
                continue
            reference = -relative[local_index, neighbor]
            displacement = increment[frame_index, neighbor] - increment[frame_index, particle]
            covariance = reference.T @ reference
            if np.linalg.cond(covariance) > 1e10:
                continue
            gradient = displacement.T @ reference @ np.linalg.inv(covariance)
            residual = displacement - reference @ gradient.T
            affine[frame_index, local_index] = gradient
            d2min[frame_index, local_index] = float(np.mean(np.sum(residual**2, axis=1)))
            valid[frame_index, local_index] = True
    return {"affine_gradient": affine, "d2min": d2min, "valid": valid}


def long_wavelength_displacement_field(
    positions: np.ndarray,
    *,
    target_indices: np.ndarray,
    box_lengths: np.ndarray,
    integer_vectors: np.ndarray,
) -> np.ndarray:
    """Reconstruct specified nonzero Fourier displacement modes at targets.

    The real field is evaluated from a positive/negative wavevector pair. It
    therefore supplies a causal, system-scale collective history variable from
    the completed preceding displacement interval.
    """

    positions = np.asarray(positions, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    integer_vectors = np.asarray(integer_vectors, dtype=int)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (at least 2 frames, particles, 3)")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= positions.shape[1]):
        raise ValueError("target_indices must select valid particles")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be positive")
    if integer_vectors.ndim != 2 or integer_vectors.shape[1] != 3 or not len(integer_vectors) or np.any(np.sum(integer_vectors**2, axis=1) == 0):
        raise ValueError("integer_vectors must be nonzero three-vectors")
    displacement = positions[1:] - positions[:-1]
    wrapped = np.mod(positions[:-1], box_lengths)
    wavevectors = 2.0 * math.pi * integer_vectors / box_lengths
    field = np.empty((len(displacement), len(target), len(wavevectors), 3), dtype=float)
    for mode_index, wavevector in enumerate(wavevectors):
        phase = wrapped @ wavevector
        amplitude = np.mean(displacement * np.exp(-1j * phase)[..., None], axis=1)
        target_phase = phase[:, target]
        field[:, :, mode_index] = 2.0 * np.real(amplitude[:, None, :] * np.exp(1j * target_phase)[..., None])
    return field


def fourier_density_current_field(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    target_indices: np.ndarray,
    box_lengths: np.ndarray,
    integer_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate translation-covariant density and current modes at tagged particles.

    For each nonzero reciprocal vector ``q`` this returns the real fields
    ``2 Re[rho_q exp(i q.r_i)]`` and ``2 Re[j_q exp(i q.r_i)]``, where
    ``rho_q = N^-1 sum_j exp(-i q.r_j)`` and
    ``j_q = N^-1 sum_j v_j exp(-i q.r_j)``.  Both are instantaneous,
    deterministic functions of the full particle phase point; no transport or
    scattering observable is used to construct them.
    """

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    integer_vectors = np.asarray(integer_vectors, dtype=int)
    if positions.ndim != 3 or positions.shape[2] != 3 or len(positions) < 1:
        raise ValueError("positions must have shape (frames, particles, 3)")
    if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
        raise ValueError("velocities must be a finite array matching positions")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= positions.shape[1]):
        raise ValueError("target_indices must select valid particles")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be positive")
    if integer_vectors.ndim != 2 or integer_vectors.shape[1] != 3 or not len(integer_vectors) or np.any(np.sum(integer_vectors**2, axis=1) == 0):
        raise ValueError("integer_vectors must be nonzero three-vectors")
    wrapped = np.mod(positions, box_lengths)
    wavevectors = 2.0 * math.pi * integer_vectors / box_lengths
    density = np.empty((len(positions), len(target), len(wavevectors)), dtype=float)
    current = np.empty((len(positions), len(target), len(wavevectors), 3), dtype=float)
    for mode_index, wavevector in enumerate(wavevectors):
        phase = wrapped @ wavevector
        phase_factor = np.exp(-1j * phase)
        density_amplitude = np.mean(phase_factor, axis=1)
        current_amplitude = np.mean(velocities * phase_factor[..., None], axis=1)
        target_phase_factor = np.exp(1j * phase[:, target])
        density[:, :, mode_index] = 2.0 * np.real(density_amplitude[:, None] * target_phase_factor)
        current[:, :, mode_index] = 2.0 * np.real(
            current_amplitude[:, None, :] * target_phase_factor[..., None]
        )
    return {"density": density, "current": current}


def symmetric_quadratic_mode_products(modes: np.ndarray) -> np.ndarray:
    """Return all unique quadratic products of real mode amplitudes.

    A real mode vector ``d_q(i)`` supplies a finite density-pair subspace via
    the products ``d_q(i)d_p(i)`` for ``q <= p``.  This remains a deterministic
    instantaneous function of the particle phase point when the input modes
    are so defined.
    """

    modes = np.asarray(modes, dtype=float)
    if modes.ndim != 3 or modes.shape[2] < 1 or not np.all(np.isfinite(modes)):
        raise ValueError("modes must be a finite (frames, samples, mode_count) array")
    first, second = np.triu_indices(modes.shape[2])
    return modes[:, :, first] * modes[:, :, second]


def propagate_free_gle_velocity_correlation(
    kernel: np.ndarray,
    *,
    frame_time: float,
    output_count: int,
    initial_correlation: float = 1.0,
) -> np.ndarray:
    """Propagate a scalar free-GLE VACF with causal trapezoid quadrature.

    For ``dC/dt = -int_0^t K(s) C(t-s) ds``, this uses the same discrete
    Volterra convention as :func:`discrete_volterra_memory_kernel`.  Kernel
    values beyond the supplied support are zero; this is a finite-support
    diagnostic, not a claim that physical memory is finite.
    """

    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 1 or not len(kernel) or not np.all(np.isfinite(kernel)):
        raise ValueError("kernel must be a nonempty finite one-vector")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be positive and finite")
    if output_count < 2 or not math.isfinite(initial_correlation) or initial_correlation == 0.0:
        raise ValueError("output_count and initial_correlation must be valid")
    correlation = np.empty(output_count, dtype=float)
    correlation[0] = initial_correlation
    for index in range(output_count - 1):
        active = min(index, len(kernel) - 1)
        convolution = 0.5 * kernel[0] * correlation[index]
        if active > 1:
            convolution += float(np.dot(kernel[1:active], correlation[index - 1 : index - active : -1]))
        if active >= 1:
            convolution += 0.5 * kernel[active] * correlation[0]
        correlation[index + 1] = correlation[index] - frame_time**2 * convolution
    return correlation


def discrete_volterra_memory_kernel(
    correlation: np.ndarray,
    *,
    frame_time: float,
    kernel_count: int,
) -> dict[str, np.ndarray]:
    """Invert a free-GLE VACF into a causal discrete memory kernel.

    The inversion is triangular under trapezoid quadrature, hence each
    ``K_n`` is fixed by ``C_0, ..., C_(n+1)`` and no macroscopic curve is
    fitted.  It is appropriate only for a stationary scalar VACF.
    """

    correlation = np.asarray(correlation, dtype=float)
    if correlation.ndim != 1 or not np.all(np.isfinite(correlation)):
        raise ValueError("correlation must be a finite one-vector")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be positive and finite")
    if kernel_count < 1 or len(correlation) < kernel_count + 1 or correlation[0] == 0.0:
        raise ValueError("correlation must include one more sample than the requested kernel")
    kernel = np.empty(kernel_count, dtype=float)
    kernel[0] = -2.0 * (correlation[1] - correlation[0]) / (frame_time**2 * correlation[0])
    for index in range(1, kernel_count):
        known = 0.5 * kernel[0] * correlation[index]
        if index > 1:
            known += float(np.dot(kernel[1:index], correlation[index - 1 : 0 : -1]))
        derivative = (correlation[index + 1] - correlation[index]) / frame_time**2
        kernel[index] = -2.0 * (derivative + known) / correlation[0]
    reconstructed = propagate_free_gle_velocity_correlation(
        kernel,
        frame_time=frame_time,
        output_count=kernel_count + 1,
        initial_correlation=float(correlation[0]),
    )
    return {"kernel": kernel, "reconstructed_correlation": reconstructed}


def discrete_mori_zwanzig_operators(
    resolved_state: np.ndarray,
    *,
    memory_order: int,
    ridge_regularization: float = 0.0,
) -> dict[str, np.ndarray]:
    """Infer finite discrete Mori--Zwanzig operators from resolved correlations.

    With column state ``g_t``, the discrete MZ correlation equation is
    ``C_(n+1)=sum_(ell=0)^n Omega_ell C_(n-ell)``.  Its triangular structure
    fixes each ``Omega_n`` from the equal-time and lagged resolved-state
    correlations.  No transport, scattering, or event-clock observable enters
    this construction.
    """

    state = np.asarray(resolved_state, dtype=float)
    if state.ndim != 3 or state.shape[2] < 1 or not np.all(np.isfinite(state)):
        raise ValueError("resolved_state must be a finite (frames, samples, dimensions) array")
    if memory_order < 0 or len(state) < memory_order + 2:
        raise ValueError("memory_order must leave one correlation lag beyond its support")
    if not math.isfinite(ridge_regularization) or ridge_regularization < 0.0:
        raise ValueError("ridge_regularization must be finite and nonnegative")
    centered = state - np.mean(state, axis=(0, 1), keepdims=True)
    dimension = state.shape[2]
    correlation = np.empty((memory_order + 2, dimension, dimension), dtype=float)
    for lag in range(memory_order + 2):
        left = centered[lag:]
        right = centered[: len(centered) - lag]
        correlation[lag] = np.einsum("tni,tnj->ij", left, right) / (left.shape[0] * left.shape[1])
    equal_time = correlation[0] + ridge_regularization * np.eye(dimension)
    if np.linalg.matrix_rank(equal_time) < dimension:
        raise ValueError("equal-time resolved covariance is singular; add ridge regularization or remove redundant variables")
    inverse_equal_time = np.linalg.inv(equal_time)
    operators = np.empty((memory_order + 1, dimension, dimension), dtype=float)
    operators[0] = correlation[1] @ inverse_equal_time
    for index in range(1, memory_order + 1):
        known = sum(operators[lag] @ correlation[index - lag] for lag in range(index))
        operators[index] = (correlation[index + 1] - known) @ inverse_equal_time
    reconstructed = np.empty_like(correlation)
    reconstructed[0] = correlation[0]
    for index in range(memory_order + 1):
        reconstructed[index + 1] = sum(operators[lag] @ reconstructed[index - lag] for lag in range(index + 1))
    return {
        "mean": np.mean(state, axis=(0, 1)),
        "correlation": correlation,
        "operators": operators,
        "reconstructed_correlation": reconstructed,
    }


def simulate_discrete_mz_empirical_innovations(
    initial_history: np.ndarray,
    operators: np.ndarray,
    innovation_pool: np.ndarray,
    *,
    output_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Propagate a finite-memory discrete MZ equation with measured innovations.

    ``initial_history`` is ordered from oldest to newest and has exactly one
    state for each supplied operator.  The first output is that newest state;
    later outputs follow ``g_(t+1)=sum_l Omega_l g_(t-l)+W_(t+1)``.
    """

    history = np.asarray(initial_history, dtype=float).copy()
    operators = np.asarray(operators, dtype=float)
    innovation_pool = np.asarray(innovation_pool, dtype=float)
    if history.ndim != 3 or operators.ndim != 3 or innovation_pool.ndim != 2:
        raise ValueError("history, operators, and innovation_pool must have ranks 3, 3, and 2")
    if history.shape[1] != len(operators) or history.shape[2] != operators.shape[1] or operators.shape[1] != operators.shape[2]:
        raise ValueError("history and square operator dimensions must align")
    if innovation_pool.shape[1] != history.shape[2] or not len(innovation_pool):
        raise ValueError("innovation_pool must be nonempty and match the resolved dimension")
    if output_count < 1 or not np.all(np.isfinite(history)) or not np.all(np.isfinite(operators)) or not np.all(np.isfinite(innovation_pool)):
        raise ValueError("states, operators, innovations, and output_count must be finite and valid")
    output = np.empty((history.shape[0], output_count, history.shape[2]), dtype=float)
    output[:, 0] = history[:, -1]
    for output_index in range(1, output_count):
        next_state = sum(
            history[:, -1 - lag] @ operators[lag].T for lag in range(len(operators))
        )
        next_state += innovation_pool[rng.integers(len(innovation_pool), size=len(history))]
        history[:, :-1] = history[:, 1:]
        history[:, -1] = next_state
        output[:, output_index] = next_state
    return output


def simulate_discrete_mz_block_innovations(
    initial_history: np.ndarray,
    operators: np.ndarray,
    innovation_series: np.ndarray,
    *,
    output_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Propagate discrete MZ dynamics with moving blocks of measured residuals.

    ``innovation_series`` has axes (time, source, resolved_dimension).  A
    source and valid time origin are redrawn at each block boundary, while the
    consecutive innovations inside a block retain their measured temporal
    order.  ``block_length=1`` is the iid empirical-innovation limit.
    """

    history = np.asarray(initial_history, dtype=float).copy()
    operators = np.asarray(operators, dtype=float)
    series = np.asarray(innovation_series, dtype=float)
    if history.ndim != 3 or operators.ndim != 3 or series.ndim != 3:
        raise ValueError("history, operators, and innovation_series must all have rank three")
    if history.shape[1] != len(operators) or history.shape[2] != operators.shape[1] or operators.shape[1] != operators.shape[2]:
        raise ValueError("history and square operator dimensions must align")
    if series.shape[2] != history.shape[2] or len(series) < 1 or series.shape[1] < 1:
        raise ValueError("innovation_series must contain time, source, and matching resolved dimensions")
    if block_length < 1 or block_length > len(series) or output_count < 1:
        raise ValueError("block_length and output_count must fit the innovation series")
    if not np.all(np.isfinite(history)) or not np.all(np.isfinite(operators)) or not np.all(np.isfinite(series)):
        raise ValueError("states, operators, and innovations must be finite")
    output = np.empty((history.shape[0], output_count, history.shape[2]), dtype=float)
    output[:, 0] = history[:, -1]
    starts = np.zeros(len(history), dtype=int)
    sources = np.zeros(len(history), dtype=int)
    for output_index in range(1, output_count):
        within_block = (output_index - 1) % block_length
        if within_block == 0:
            starts = rng.integers(len(series) - block_length + 1, size=len(history))
            sources = rng.integers(series.shape[1], size=len(history))
        next_state = sum(
            history[:, -1 - lag] @ operators[lag].T for lag in range(len(operators))
        )
        next_state += series[starts + within_block, sources]
        history[:, :-1] = history[:, 1:]
        history[:, -1] = next_state
        output[:, output_index] = next_state
    return output


def time_split_collective_velocity_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    neighbor_velocity: np.ndarray,
    *,
    train_stop: int,
) -> dict[str, float]:
    """Fit a one-step center GLE with an explicit local environment velocity.

    The prediction is conditional on the current tagged-center velocity and
    the current neighbor velocity.  It tests whether this minimal collective
    Markov embedding turns the projected random force approximately Gaussian.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    neighbor_velocity = np.asarray(neighbor_velocity, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if neighbor_velocity.shape != (centers.shape[0] - 1, centers.shape[1], 3):
        raise ValueError("neighbor_velocity must contain one vector per center interval")
    if not (3 < train_stop < centers.shape[0] - 3):
        raise ValueError("train_stop must leave at least two interval pairs in each segment")

    velocity = centers[1:] - centers[:-1]
    pair_valid = valid[:-2] & valid[1:-1] & valid[2:]
    pair_valid &= np.isfinite(neighbor_velocity[:-1]).all(axis=2)
    pair_valid &= np.isfinite(velocity[:-1]).all(axis=2) & np.isfinite(velocity[1:]).all(axis=2)
    predictor_velocity = velocity[:-1]
    predictor_neighbor = neighbor_velocity[:-1]
    response = velocity[1:]
    train_mask = pair_valid[: train_stop - 2]
    held_mask = pair_valid[train_stop - 1 :]
    if not np.any(train_mask) or not np.any(held_mask):
        raise ValueError("train and held-out segments require valid velocity pairs")

    x_train = np.column_stack(
        [predictor_velocity[: train_stop - 2][train_mask].reshape(-1), predictor_neighbor[: train_stop - 2][train_mask].reshape(-1)]
    )
    y_train = response[: train_stop - 2][train_mask].reshape(-1)
    coefficients, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    ar_coefficient = float(np.dot(x_train[:, 0], y_train) / np.dot(x_train[:, 0], x_train[:, 0]))

    x_held = np.column_stack(
        [predictor_velocity[train_stop - 1 :][held_mask].reshape(-1), predictor_neighbor[train_stop - 1 :][held_mask].reshape(-1)]
    )
    y_held = response[train_stop - 1 :][held_mask].reshape(-1)
    prediction = x_held @ coefficients
    ar_prediction = x_held[:, 0] * ar_coefficient
    total = float(np.sum((y_held - np.mean(y_held)) ** 2))
    r_squared = 1.0 - float(np.sum((y_held - prediction) ** 2)) / total if total > 0.0 else math.nan
    ar_r_squared = 1.0 - float(np.sum((y_held - ar_prediction) ** 2)) / total if total > 0.0 else math.nan
    residual = (y_held - prediction).reshape(-1, 3)
    return {
        "center_velocity_coefficient": float(coefficients[0]),
        "collective_velocity_coefficient": float(coefficients[1]),
        "heldout_collective_r_squared": float(r_squared),
        "heldout_ar_only_r_squared": float(ar_r_squared),
        "heldout_collective_r_squared_gain": float(r_squared - ar_r_squared),
        "collective_residual_ngp": float(_ngp_3d(residual)),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def time_split_affine_environment_embedding_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    affine_gradient: np.ndarray,
    d2min: np.ndarray,
    affine_valid: np.ndarray,
    *,
    train_stop: int,
    ridge_regularization: float = 1e-6,
) -> dict[str, float]:
    """Test a causal local-affine environment embedding for center velocity."""

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    affine_gradient = np.asarray(affine_gradient, dtype=float)
    d2min = np.asarray(d2min, dtype=float)
    affine_valid = np.asarray(affine_valid, dtype=bool)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if affine_gradient.shape != (centers.shape[0] - 1, centers.shape[1], 3, 3):
        raise ValueError("affine_gradient must align as (center intervals, particles, 3, 3)")
    if d2min.shape != affine_gradient.shape[:2] or affine_valid.shape != d2min.shape:
        raise ValueError("d2min and affine_valid must align with affine_gradient")
    if not (4 < train_stop < centers.shape[0] - 3) or ridge_regularization < 0.0:
        raise ValueError("train_stop and ridge_regularization must be valid")

    velocity = centers[1:] - centers[:-1]
    index = np.arange(1, len(velocity) - 1)
    usable = valid[index] & valid[index + 1] & valid[index + 2]
    usable &= affine_valid[index - 1] & np.isfinite(affine_gradient[index - 1]).all(axis=(2, 3))
    usable &= np.isfinite(d2min[index - 1]) & (d2min[index - 1] > 0.0)
    usable &= np.isfinite(velocity[index]).all(axis=2) & np.isfinite(velocity[index + 1]).all(axis=2)
    train_selector = index < train_stop - 2
    held_selector = index >= train_stop - 1
    train_mask = usable[train_selector]
    held_mask = usable[held_selector]
    if not np.any(train_mask) or not np.any(held_mask):
        raise ValueError("both time segments require valid affine environment samples")

    def features(selector: np.ndarray, mask: np.ndarray) -> np.ndarray:
        current_velocity = velocity[index[selector]][mask].reshape(-1)
        affine = affine_gradient[index[selector] - 1][mask].reshape(-1, 3)
        d2 = np.log(d2min[index[selector] - 1][mask])
        return np.column_stack([current_velocity, affine, np.repeat(d2, 3)])

    x_train = features(train_selector, train_mask)
    y_train = velocity[index[train_selector] + 1][train_mask].reshape(-1)
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized_train = (x_train - mean) / scale
    penalty = ridge_regularization * np.eye(standardized_train.shape[1])
    coefficients = np.linalg.solve(standardized_train.T @ standardized_train + penalty, standardized_train.T @ y_train)
    ar_coefficient = float(np.dot(x_train[:, 0], y_train) / np.dot(x_train[:, 0], x_train[:, 0]))

    x_held = features(held_selector, held_mask)
    y_held = velocity[index[held_selector] + 1][held_mask].reshape(-1)
    prediction = ((x_held - mean) / scale) @ coefficients
    ar_prediction = x_held[:, 0] * ar_coefficient
    total = float(np.sum((y_held - np.mean(y_held)) ** 2))
    affine_r_squared = 1.0 - float(np.sum((y_held - prediction) ** 2)) / total if total > 0.0 else math.nan
    ar_r_squared = 1.0 - float(np.sum((y_held - ar_prediction) ** 2)) / total if total > 0.0 else math.nan
    residual = (y_held - prediction).reshape(-1, 3)
    return {
        "center_velocity_coefficient": float(coefficients[0] / scale[0]),
        "heldout_affine_r_squared": float(affine_r_squared),
        "heldout_ar_only_r_squared": float(ar_r_squared),
        "heldout_affine_r_squared_gain": float(affine_r_squared - ar_r_squared),
        "affine_residual_ngp": float(_ngp_3d(residual)),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def time_split_collective_field_embedding_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    collective_field: np.ndarray,
    *,
    train_stop: int,
    ridge_regularization: float = 1e-6,
) -> dict[str, float]:
    """Test whether a causal low-wavevector displacement field closes center velocity.

    The field is reconstructed from the completed interval before the predictor
    center velocity.  Each Fourier mode supplies one vector component aligned
    with the predicted center-velocity component, preserving translational and
    rotational covariance without fitting to any macroscopic observable.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    collective_field = np.asarray(collective_field, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if collective_field.ndim != 4 or collective_field.shape[:2] != (centers.shape[0] - 1, centers.shape[1]) or collective_field.shape[3] != 3:
        raise ValueError("collective_field must align as (center intervals, particles, modes, 3)")
    if not (4 < train_stop < centers.shape[0] - 3) or ridge_regularization < 0.0:
        raise ValueError("train_stop and ridge_regularization must be valid")

    velocity = centers[1:] - centers[:-1]
    index = np.arange(1, len(velocity) - 1)
    usable = valid[index] & valid[index + 1] & valid[index + 2]
    usable &= np.isfinite(velocity[index]).all(axis=2) & np.isfinite(velocity[index + 1]).all(axis=2)
    usable &= np.isfinite(collective_field[index - 1]).all(axis=(2, 3))
    train_selector = index < train_stop - 2
    held_selector = index >= train_stop - 1
    train_mask = usable[train_selector]
    held_mask = usable[held_selector]
    if not np.any(train_mask) or not np.any(held_mask):
        raise ValueError("both time segments require valid collective field samples")

    def features(selector: np.ndarray, mask: np.ndarray) -> np.ndarray:
        current_velocity = velocity[index[selector]][mask].reshape(-1)
        field = collective_field[index[selector] - 1][mask]
        mode_components = field.transpose(0, 2, 1).reshape(-1, field.shape[1])
        return np.column_stack([current_velocity, mode_components])

    x_train = features(train_selector, train_mask)
    y_train = velocity[index[train_selector] + 1][train_mask].reshape(-1)
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized_train = (x_train - mean) / scale
    penalty = ridge_regularization * np.eye(standardized_train.shape[1])
    coefficients = np.linalg.solve(standardized_train.T @ standardized_train + penalty, standardized_train.T @ y_train)
    ar_coefficient = float(np.dot(x_train[:, 0], y_train) / np.dot(x_train[:, 0], x_train[:, 0]))

    x_held = features(held_selector, held_mask)
    y_held = velocity[index[held_selector] + 1][held_mask].reshape(-1)
    prediction = ((x_held - mean) / scale) @ coefficients
    ar_prediction = x_held[:, 0] * ar_coefficient
    total = float(np.sum((y_held - np.mean(y_held)) ** 2))
    field_r_squared = 1.0 - float(np.sum((y_held - prediction) ** 2)) / total if total > 0.0 else math.nan
    ar_r_squared = 1.0 - float(np.sum((y_held - ar_prediction) ** 2)) / total if total > 0.0 else math.nan
    residual = (y_held - prediction).reshape(-1, 3)
    return {
        "center_velocity_coefficient": float(coefficients[0] / scale[0]),
        "collective_field_coefficient_norm": float(np.linalg.norm(coefficients[1:] / scale[1:])),
        "heldout_collective_field_r_squared": float(field_r_squared),
        "heldout_ar_only_r_squared": float(ar_r_squared),
        "heldout_collective_field_r_squared_gain": float(field_r_squared - ar_r_squared),
        "collective_field_residual_ngp": float(_ngp_3d(residual)),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def time_split_shell_response_embedding_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    shell_response: np.ndarray,
    shell_valid: np.ndarray,
    *,
    train_stop: int,
) -> dict[str, float]:
    """Test a causal shell-resolved microscopic response-history embedding.

    The response field from the completed preceding interval is decomposed
    into fixed radial shells.  This preserves spatial structure that is lost
    when all neighbor contributions are summed into a single vector.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    shell_response = np.asarray(shell_response, dtype=float)
    shell_valid = np.asarray(shell_valid, dtype=bool)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if shell_response.ndim != 4 or shell_response.shape[:2] != (centers.shape[0] - 1, centers.shape[1]) or shell_response.shape[3] != 3:
        raise ValueError("shell_response must align as (center intervals, particles, shells, 3)")
    if shell_valid.shape != shell_response.shape[:3]:
        raise ValueError("shell_valid must align with shell_response")
    if not (4 < train_stop < centers.shape[0] - 3):
        raise ValueError("train_stop must leave shell-response samples in both segments")

    velocity = centers[1:] - centers[:-1]
    index = np.arange(1, len(velocity) - 1)
    usable = valid[index] & valid[index + 1] & valid[index + 2]
    usable &= np.all(shell_valid[index - 1], axis=2)
    usable &= np.isfinite(velocity[index]).all(axis=2) & np.isfinite(velocity[index + 1]).all(axis=2)
    usable &= np.isfinite(shell_response[index - 1]).all(axis=(2, 3))
    train_selector = index < train_stop - 2
    held_selector = index >= train_stop - 1
    train_mask = usable[train_selector]
    held_mask = usable[held_selector]
    if not np.any(train_mask) or not np.any(held_mask):
        raise ValueError("both time segments require valid shell-response samples")

    def features(selector: np.ndarray, mask: np.ndarray) -> np.ndarray:
        terms = [velocity[index[selector]][mask].reshape(-1)]
        for shell in range(shell_response.shape[2]):
            terms.append(shell_response[index[selector] - 1, :, shell][mask].reshape(-1))
        return np.column_stack(terms)

    x_train = features(train_selector, train_mask)
    y_train = velocity[index[train_selector] + 1][train_mask].reshape(-1)
    coefficients, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    ar_coefficient = float(np.dot(x_train[:, 0], y_train) / np.dot(x_train[:, 0], x_train[:, 0]))
    x_held = features(held_selector, held_mask)
    y_held = velocity[index[held_selector] + 1][held_mask].reshape(-1)
    prediction = x_held @ coefficients
    ar_prediction = x_held[:, 0] * ar_coefficient
    total = float(np.sum((y_held - np.mean(y_held)) ** 2))
    shell_r_squared = 1.0 - float(np.sum((y_held - prediction) ** 2)) / total if total > 0.0 else math.nan
    ar_r_squared = 1.0 - float(np.sum((y_held - ar_prediction) ** 2)) / total if total > 0.0 else math.nan
    result = {
        "center_velocity_coefficient": float(coefficients[0]),
        "heldout_shell_r_squared": float(shell_r_squared),
        "heldout_ar_only_r_squared": float(ar_r_squared),
        "heldout_shell_r_squared_gain": float(shell_r_squared - ar_r_squared),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for shell, coefficient in enumerate(coefficients[1:]):
        result[f"shell_{shell}_coefficient"] = float(coefficient)
    return result
