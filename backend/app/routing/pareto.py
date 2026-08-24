"""The demo money shot: a family of routes trading distance off against heat
exposure, and a plain-English comparison between the extremes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.routing.router import RouteResult, route

logger = logging.getLogger(__name__)

DEFAULT_LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def compute_route_family(
    start: tuple[float, float],
    end: tuple[float, float],
    graph,
    grid,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    **route_kwargs,
) -> list[RouteResult]:
    """One route per lambda, deduplicated by identical node paths. Labels the
    extremes SHORTEST (lambda=min) / COOLEST (lambda=max) and the middle lambda
    value BALANCED, by GROUP membership rather than "first lambda wins" --
    several lambdas often collapse onto the same path (few genuinely distinct
    routes exist between two points), and if lambda=max shares a path with
    lambda=mid, that path must still be labelled COOLEST, not silently
    demoted to BALANCED just because mid was processed first.
    """
    sorted_lambdas = sorted(lambdas)
    if not sorted_lambdas:
        raise ValueError("lambdas must be non-empty")
    min_lambda, max_lambda = sorted_lambdas[0], sorted_lambdas[-1]
    mid_lambda = sorted_lambdas[len(sorted_lambdas) // 2]

    all_results = [route(start, end, lam, graph, grid, **route_kwargs) for lam in sorted_lambdas]

    groups: dict[tuple[int, ...], list[tuple[float, RouteResult]]] = {}
    order: list[tuple[int, ...]] = []
    for lam, result in zip(sorted_lambdas, all_results):
        path_key = tuple(result.node_path)
        if path_key not in groups:
            groups[path_key] = []
            order.append(path_key)
        groups[path_key].append((lam, result))

    deduped: list[RouteResult] = []
    for path_key in order:
        members = groups[path_key]
        member_lambdas = {lam for lam, _ in members}
        representative = members[0][1]  # keep the result from the first lambda that produced this path

        if min_lambda in member_lambdas:
            representative.label = "SHORTEST"
        elif max_lambda in member_lambdas:
            representative.label = "COOLEST"
        elif mid_lambda in member_lambdas:
            representative.label = "BALANCED"

        if len(members) > 1:
            logger.info(
                "compute_route_family: lambdas %s all produce the same path -- deduplicated to one entry (label=%s)",
                sorted(member_lambdas),
                representative.label,
            )
        deduped.append(representative)

    return deduped


@dataclass
class RouteComparison:
    extra_distance_m: float
    extra_distance_pct: float
    extra_duration_s: float
    dose_reduction_degC_s: float
    dose_reduction_pct: float
    mean_wbgt_delta_c: float
    same_path: bool
    summary: str


def compare(shortest: RouteResult, coolest: RouteResult) -> RouteComparison:
    """Compares the SHORTEST (lambda=0) and COOLEST (lambda=1) routes. If they're
    the same path, says so explicitly rather than fabricating a nonexistent tradeoff.
    """
    if shortest.node_path == coolest.node_path:
        return RouteComparison(
            extra_distance_m=0.0,
            extra_distance_pct=0.0,
            extra_duration_s=0.0,
            dose_reduction_degC_s=0.0,
            dose_reduction_pct=0.0,
            mean_wbgt_delta_c=0.0,
            same_path=True,
            summary="The shortest and coolest routes are the SAME path -- there is no heat/distance tradeoff to make between these two points right now.",
        )

    extra_distance_m = coolest.total_distance_m - shortest.total_distance_m
    extra_distance_pct = (extra_distance_m / shortest.total_distance_m * 100) if shortest.total_distance_m else 0.0
    extra_duration_s = coolest.total_duration_s - shortest.total_duration_s

    dose_reduction_degC_s = shortest.total_heat_dose_degC_s - coolest.total_heat_dose_degC_s
    dose_reduction_pct = (
        (dose_reduction_degC_s / shortest.total_heat_dose_degC_s * 100) if shortest.total_heat_dose_degC_s else 0.0
    )
    mean_wbgt_delta_c = coolest.mean_wbgt_c - shortest.mean_wbgt_c

    summary = (
        f"{extra_distance_m:.0f} m longer ({extra_distance_pct:+.0f}%), "
        f"{-mean_wbgt_delta_c:.1f} C cooler on average, "
        f"{dose_reduction_pct:.0f}% less heat exposure"
    )

    return RouteComparison(
        extra_distance_m=extra_distance_m,
        extra_distance_pct=extra_distance_pct,
        extra_duration_s=extra_duration_s,
        dose_reduction_degC_s=dose_reduction_degC_s,
        dose_reduction_pct=dose_reduction_pct,
        mean_wbgt_delta_c=mean_wbgt_delta_c,
        same_path=False,
        summary=summary,
    )
