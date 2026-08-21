#!/usr/bin/env python3
"""Analyze the non-mutating NVMAI v4.3 predictive-prefetch trace.

The trace is collected with ``NVMAI_PREFETCH_TRACE=/path/to/trace.jsonl``.
This tool intentionally models a cheap, online transition-prior predictor;
it never changes model routing or runtime cache state. Its purpose is to
decide whether a real, bounded staging-ring implementation has enough useful
nonresident-miss recall to warrant integration.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Observation:
    position: int
    layer: int
    experts: tuple[int, ...]
    misses: frozenset[int]
    resident: frozenset[int]
    next_layer_prediction: tuple[int, ...] = ()


@dataclass
class Counters:
    links: int = 0
    actual_misses: int = 0
    predicted: int = 0
    useful: int = 0
    overlap: int = 0
    actual_experts: int = 0

    def report(self) -> dict[str, float | int]:
        return {
            "links": self.links,
            "actual_misses": self.actual_misses,
            "predicted_nonresident": self.predicted,
            "useful_prefetches": self.useful,
            "raw_topk_agreement": self.overlap / self.actual_experts
            if self.actual_experts else 0.0,
            "actual_miss_recall": self.useful / self.actual_misses
            if self.actual_misses else 0.0,
            "prefetch_precision": self.useful / self.predicted
            if self.predicted else 0.0,
        }


def load_trace(path: pathlib.Path) -> list[Observation]:
    observations: list[Observation] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            observations.append(Observation(
                position=int(value["position"]),
                layer=int(value["layer"]),
                experts=tuple(int(expert) for expert in value["experts"]),
                misses=frozenset(int(expert) for expert in value["misses"]),
                resident=frozenset(int(expert) for expert in value["resident"]),
                next_layer_prediction=tuple(
                    int(expert) for expert in value.get("next_layer_prediction", [])),
            ))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{number}: invalid trace observation: {error}") from error
    return sorted(observations, key=lambda item: (item.position, item.layer))


def ranked_transition_prediction(
    counts: dict[int, dict[int, collections.Counter[int]]],
    source: Observation,
    top_m: int,
) -> list[int]:
    scores: collections.Counter[int] = collections.Counter()
    for expert in source.experts:
        scores.update(counts[source.layer][expert])
    return [expert for expert, _ in scores.most_common(top_m)]


def analyze(observations: Iterable[Observation], top_m: int,
            predictor: str = "transition") -> dict[str, object]:
    """Evaluate a trace-only L -> L+1 predictor without future leakage."""
    by_position: dict[int, dict[int, Observation]] = collections.defaultdict(dict)
    for observation in observations:
        by_position[observation.position][observation.layer] = observation

    counts: dict[int, dict[int, collections.Counter[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter))
    overall = Counters()
    per_link: dict[int, Counters] = collections.defaultdict(Counters)
    for position in sorted(by_position):
        layers = by_position[position]
        for layer in sorted(layers):
            source = layers[layer]
            target = layers.get(layer + 1)
            if target is None:
                continue
            candidates = (
                list(source.next_layer_prediction[:top_m])
                if predictor == "recorded"
                else ranked_transition_prediction(counts, source, top_m)
            )
            issued = [expert for expert in candidates if expert not in target.resident]
            useful = set(issued).intersection(target.misses)
            overlap = set(candidates).intersection(target.experts)
            for bucket in (overall, per_link[layer]):
                bucket.links += 1
                bucket.actual_misses += len(target.misses)
                bucket.predicted += len(issued)
                bucket.useful += len(useful)
                bucket.overlap += len(overlap)
                bucket.actual_experts += len(target.experts)
            for source_expert in source.experts:
                counts[layer][source_expert].update(target.experts)

    return {
        "predictor": predictor,
        "top_m": top_m,
        "overall": overall.report(),
        "per_layer_link": {
            str(layer): bucket.report() for layer, bucket in sorted(per_link.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--predictor", choices=("recorded", "transition"),
                        default="recorded")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.top_m < 1:
        parser.error("--top-m must be positive")
    report = analyze(load_trace(args.trace), args.top_m, args.predictor)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
