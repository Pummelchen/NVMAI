import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("nvmai_prefetch_trace_analysis.py")
SPEC = importlib.util.spec_from_file_location("prefetch_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrefetchTraceAnalysisTests(unittest.TestCase):
    def test_online_transition_prior_reports_nonresident_miss_recall(self) -> None:
        observations = [
            MODULE.Observation(0, 0, (1, 2), frozenset({1, 2}), frozenset(), (7, 8)),
            MODULE.Observation(0, 1, (7, 8), frozenset({7, 8}), frozenset()),
            MODULE.Observation(1, 0, (1, 2), frozenset(), frozenset({1, 2}), (7, 9)),
            MODULE.Observation(1, 1, (7, 9), frozenset({7, 9}), frozenset()),
        ]

        report = MODULE.analyze(observations, top_m=2, predictor="recorded")
        overall = report["overall"]
        self.assertEqual(overall["links"], 2)
        self.assertEqual(overall["actual_misses"], 4)
        self.assertEqual(overall["useful_prefetches"], 4)
        self.assertEqual(overall["actual_miss_recall"], 1.0)
        self.assertEqual(overall["prefetch_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
