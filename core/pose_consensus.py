"""Revisable, order-independent multi-view pose support (no GT inputs)."""
from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np


@dataclass
class ConsensusResult:
    selected: list[int]
    pose: gtsam.Pose3
    ambiguous: bool
    pair_consistent: np.ndarray


def robust_pose_center(poses: list[gtsam.Pose3], covariances: list[np.ndarray],
                       huber_k: float) -> gtsam.Pose3:
    """Small canonical SE(3) IRLS solve used only to initialize/select factors."""
    order = sorted(range(len(poses)), key=lambda i: tuple(poses[i].matrix().ravel()))
    precisions = [np.linalg.inv(c) for c in covariances]
    # A medoid avoids averaging opposing rotation hypotheses at initialization.
    def cost(i):
        return sum(min(np.linalg.norm(gtsam.Pose3.Logmap(poses[i].between(poses[j]))), 1.0)
                   for j in order)
    center = poses[min(order, key=cost)]
    for _ in range(12):
        info, rhs = np.zeros((6, 6)), np.zeros(6)
        for i in order:
            r = gtsam.Pose3.Logmap(center.between(poses[i]))
            length = np.sqrt(max(float(r @ precisions[i] @ r), 0.0))
            weight = min(1.0, huber_k / max(length, 1e-12)) if huber_k > 0 else 1.0
            info += weight * precisions[i]
            rhs += weight * (precisions[i] @ r)
        delta = np.linalg.solve(info, rhs)
        center = center.compose(gtsam.Pose3.Expmap(delta))
        if np.linalg.norm(delta) < 1e-7:
            break
    return center


def select_pose_consensus(poses: list[gtsam.Pose3], covariances: list[np.ndarray], *,
                          trans_gate_m: float, rot_gate_deg: float,
                          nis_threshold: float | None, huber_k: float,
                          prediction: gtsam.Pose3 | None = None,
                          qualities: list[float] | None = None,
                          quality_override_ratio: float = 1.5) -> ConsensusResult:
    n = len(poses)
    if not n:
        raise ValueError("Consensus requires at least one observation")
    distances = np.zeros((n, n))
    compatible = np.eye(n, dtype=bool)
    for i in range(n):
        for j in range(i):
            delta = poses[i].between(poses[j])
            r = gtsam.Pose3.Logmap(delta)
            # Both observation covariances are in Pose3 body tangent order.
            # Linearize Log(T_i^-1 T_j), rather than add world translations
            # to object-local covariances without transforming the Jacobians.
            h1, h2 = np.empty((6, 6), order="F"), np.empty((6, 6), order="F")
            poses[i].between(poses[j], h1, h2)
            log_jac = gtsam.Pose3.LogmapDerivative(delta)
            a, b = log_jac @ h1, log_jac @ h2
            covariance = a @ covariances[i] @ a.T + b @ covariances[j] @ b.T
            nis = max(float(r @ np.linalg.solve(covariance, r)), 0.0)
            distances[i, j] = distances[j, i] = nis
            agree = (np.linalg.norm(delta.translation()) <= trans_gate_m
                     and np.degrees(np.linalg.norm(r[:3])) <= rot_gate_deg
                     and (nis_threshold is None or nis <= nis_threshold))
            compatible[i, j] = compatible[j, i] = agree
    canonical = lambda i: tuple(poses[i].matrix().ravel()) + tuple(covariances[i].ravel())
    # Greedy complete-link candidates for every seed: bounded O(N^3), supports
    # arbitrary camera counts and cannot join outliers through a similarity chain.
    candidates = set()
    for seed in sorted(range(n), key=canonical):
        cluster = [seed]
        for other in sorted((j for j in range(n) if j != seed), key=lambda j: (distances[seed, j], canonical(j))):
            if all(compatible[other, k] for k in cluster):
                cluster.append(other)
        candidates.add(tuple(sorted(cluster)))
    size = max(map(len, candidates))
    contenders = [c for c in candidates if len(c) == size]
    def rank(cluster):
        if prediction is not None:
            temporal = np.median([np.linalg.norm(gtsam.Pose3.Logmap(prediction.between(poses[i]))) for i in cluster])
        else:
            temporal = 0.0
        uncertainty = sum(float(np.log(np.linalg.det(covariances[i]))) for i in cluster)
        dispersion = sum(distances[i, j] for i in cluster for j in cluster)
        return (temporal, uncertainty, dispersion, tuple(sorted(canonical(i) for i in cluster)))
    chosen = min(contenders, key=rank)
    # A large but mutually-consistent cluster can still be a correlated wrong
    # hypothesis.  When an independently measured RGB-D/CAD quality term is
    # available, permit a substantially better smaller candidate to override
    # cardinality.  This remains opt-in because the ratio needs calibration.
    if qualities is not None and len(qualities) == n and quality_override_ratio > 1.0:
        finite_quality = np.asarray(qualities, dtype=float)
        if np.isfinite(finite_quality).all() and np.all(finite_quality >= 0.0):
            # Zero depth inliers is valid negative evidence, not missing data.
            # Do not let one fully occluded view disable all other quality
            # comparisons. The floor only makes log(0) finite; it is not a
            # detection threshold and never removes a raw observation.
            log_quality = np.log(np.maximum(finite_quality, 1e-12))
            minimum_size = max(1, size - 1)
            quality_candidates = [c for c in candidates if len(c) >= minimum_size]
            def quality_rank(cluster):
                return float(np.mean(log_quality[list(cluster)]))
            quality_best = min(quality_candidates, key=lambda c: (-quality_rank(c), rank(c)))
            if (quality_rank(quality_best) - quality_rank(chosen)
                    >= np.log(float(quality_override_ratio))):
                chosen = quality_best

    selected = sorted(chosen, key=canonical)
    center = robust_pose_center([poses[i] for i in selected], [covariances[i] for i in selected], huber_k)
    return ConsensusResult(selected, center, len(contenders) > 1, compatible)


class SensorConsistencyHistory:
    """One replaceable vote per object/state/sensor, never per publication."""

    def __init__(self, failures: int = 2, successes: int = 3, window: int = 32):
        self.failures, self.successes, self.window = failures, successes, window
        self.states: dict[tuple[str, str], dict[int, bool | None]] = {}

    def update(self, object_name: str, sensor: str, step: int, good: bool | None) -> str:
        history = self.states.setdefault((object_name, sensor), {})
        history[step] = good
        for old in sorted(history)[:-self.window]:
            del history[old]
        votes = [history[k] for k in sorted(history, reverse=True)]
        if len(votes) >= self.failures and all(v is False for v in votes[:self.failures]):
            return "Unsafe"
        if len(votes) >= self.successes and all(v is True for v in votes[:self.successes]):
            return "Consistent"
        return "Unverified"
