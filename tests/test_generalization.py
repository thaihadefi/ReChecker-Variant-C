import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config.config import ExperimentConfig
from data.audit_dappscan import Finding, finding_visibility, locate_findings, parse_line_range
from data.build_smartbugs_dataset import classify
from data.types import GadgetRecord
from experiments.distribution_shift import (
    em_prior_adjust,
    evaluate_strategies,
    rate_matched_threshold,
    resample_to_prevalence,
)
from experiments.run_cross_dataset import TrainingSetup, _parse_test_files, prepare_records
from representation.pruning import (
    UNK_TOKEN,
    abstract_identifiers,
    abstract_token,
    deduplicate_groups,
    fit_group_vocabulary,
    prune_vocabulary,
    sample_group_fraction,
)


def _record(sample_id, label, tokens, group_id=None, c_tokens=("c",)):
    return GadgetRecord(
        sample_id=sample_id,
        label=label,
        w_tokens=tuple(tokens),
        c_tokens=tuple(c_tokens),
        group_id=group_id or sample_id,
    )


class PruningTests(unittest.TestCase):
    def test_abstract_token_drops_indices_and_literals(self):
        self.assertEqual(abstract_token("VAR12"), "VAR")
        self.assertEqual(abstract_token("FUN3"), "FUN")
        self.assertEqual(abstract_token("1e18"), "NUM")
        self.assertEqual(abstract_token("0xFF"), "NUM")
        self.assertEqual(abstract_token("call"), "call")
        self.assertEqual(abstract_token("VARIABLE"), "VARIABLE")

    def test_abstraction_merges_groups_that_become_identical(self):
        records = abstract_identifiers([
            _record("a", 1, ["VAR1", "=", "1"]),
            _record("b", 1, ["VAR7", "=", "2"]),
        ])
        self.assertEqual(records[0].group_id, records[1].group_id)
        self.assertEqual(records[0].w_tokens, ("VAR", "=", "NUM"))

    def test_deduplicate_keeps_first_of_each_group(self):
        records = [_record("a", 1, ["x"], "g1"), _record("b", 1, ["x"], "g1"), _record("c", 0, ["y"], "g2")]
        self.assertEqual([r.sample_id for r in deduplicate_groups(records)], ["a", "c"])

    def test_deduplicate_rejects_conflicting_labels(self):
        with self.assertRaises(ValueError):
            deduplicate_groups([_record("a", 1, ["x"], "g"), _record("b", 0, ["x"], "g")])

    def test_group_vocabulary_counts_groups_not_copies(self):
        records = [_record(f"d{i}", 1, ["rare"], "g1") for i in range(5)]
        records += [_record("e", 0, ["common"], "g2"), _record("f", 0, ["common"], "g3")]
        vocabulary = fit_group_vocabulary(records, min_groups=2)
        self.assertIn("common", vocabulary)
        self.assertNotIn("rare", vocabulary)
        pruned = prune_vocabulary(records, vocabulary)
        self.assertEqual(pruned[0].w_tokens, (UNK_TOKEN,))
        self.assertEqual(pruned[0].group_id, "g1")

    def test_sample_group_fraction_keeps_whole_groups_per_label(self):
        records = [_record(f"p{i}", 1, ["x"], f"pg{i // 2}") for i in range(8)]
        records += [_record(f"n{i}", 0, ["y"], f"ng{i // 2}") for i in range(8)]
        kept = sample_group_fraction(records, 0.5, seed=3)
        groups = {r.group_id for r in kept}
        self.assertEqual(len([g for g in groups if g.startswith("pg")]), 2)
        self.assertEqual(len([g for g in groups if g.startswith("ng")]), 2)
        self.assertEqual(len(kept), 8)
        self.assertEqual(sample_group_fraction(records, 1.0, seed=3), records)


class PrepareRecordsTests(unittest.TestCase):
    def test_vocabulary_is_fitted_on_fit_partition_only(self):
        train = []
        for group in range(10):
            label = group % 2
            train += [_record(f"s{group}-{k}", label, ["shared", f"own{group}"], f"g{group}") for k in range(2)]
        tests = {"t": [_record("t0", 1, ["shared", "own0", "testonly"])]}
        config = ExperimentConfig(validation_folds=2, seed=0)
        fit, val, pruned_tests = prepare_records(
            train, tests, TrainingSetup(dedup=True, min_token_groups=2), config, seed=0
        )
        self.assertEqual(len(fit) + len(val), 10)
        self.assertEqual(pruned_tests["t"][0].w_tokens, ("shared", UNK_TOKEN, UNK_TOKEN))

    def test_parse_test_files(self):
        self.assertEqual(_parse_test_files("a=x.txt,b=y.txt"), {"a": Path("x.txt"), "b": Path("y.txt")})
        with self.assertRaises(ValueError):
            _parse_test_files("broken")


class DistributionShiftTests(unittest.TestCase):
    def test_resample_hits_requested_prevalence(self):
        labels = np.array([1] * 100 + [0] * 800)
        rng = np.random.default_rng(0)
        for prevalence in (0.1, 0.5, 2 / 3):
            index = resample_to_prevalence(labels, prevalence, rng)
            self.assertAlmostEqual(labels[index].mean(), prevalence, delta=0.01)
            self.assertEqual(len(set(index)), len(index))

    def test_rate_matched_threshold_flags_requested_share(self):
        scores = np.linspace(0, 1, 100)
        threshold = rate_matched_threshold(scores, 0.2)
        self.assertEqual(int((scores >= threshold).sum()), 20)
        self.assertEqual(int((scores >= rate_matched_threshold(scores, 0.0)).sum()), 0)

    def test_em_recovers_shifted_prior(self):
        rng = np.random.default_rng(1)
        # Calibrated posteriors under a 50% source prior, evaluated on a 20% target.
        labels = (rng.random(20000) < 0.2).astype(int)
        score = np.where(labels == 1, 1.5, -1.5) + rng.normal(0, 1, len(labels))
        # Class-conditionals N(+-1.5, 1) give log-likelihood ratio 3 * score.
        posterior = 1 / (1 + np.exp(-3 * score))
        _, prior = em_prior_adjust(posterior, source_prior=0.5)
        self.assertAlmostEqual(prior, 0.2, delta=0.04)

    def test_strategies_share_rank_metrics(self):
        rng = np.random.default_rng(2)
        val_y = rng.integers(0, 2, 200)
        val_p = np.clip(val_y * 0.3 + rng.random(200) * 0.7, 0, 1)
        test_y = rng.integers(0, 2, 300)
        test_p = np.clip(test_y * 0.2 + rng.random(300) * 0.5, 0, 1)
        results = evaluate_strategies(val_y, val_p, test_y, test_p)
        self.assertEqual(len({round(m["roc_auc"], 12) for m in results.values()}), 1)
        oracle = results["oracle_prior"]
        self.assertAlmostEqual(oracle["predicted_positive_rate"], test_y.mean(), delta=0.01)


class AuditTests(unittest.TestCase):
    def test_parse_line_range(self):
        self.assertEqual(parse_line_range("L182-L240"), (182, 240))
        self.assertEqual(parse_line_range("L79"), (79, 79))
        self.assertIsNone(parse_line_range("N/A"))

    def test_locate_findings_skips_inserted_annotations(self):
        source = "a\n// SWC-107-Reentrancy: L2\nb\nc\n"
        self.assertEqual(locate_findings(source, [Finding("f", 2, 3)]), [Finding("f", 3, 4)])

    def test_finding_visibility_respects_region_budgets(self):
        source = "\n".join([
            "pragma solidity ^0.8.0;",
            "contract A {",
            "function first() public {",
            "x = 1;",
            "}",
            "function second() public {",
            "msg.sender.call(y);",
            "}",
        ])
        early = finding_visibility(source, [Finding("first", 4, 4)], max_len=500)
        self.assertTrue(early["in_first_function"])
        self.assertTrue(early["visible_region_prefix"])
        late = finding_visibility(source, [Finding("second", 7, 7)], max_len=12, w_ratio=0.8)
        self.assertFalse(late["in_first_function"])
        self.assertTrue(late["extracted_at_all"])
        self.assertFalse(late["visible_region_prefix"])
        preamble = finding_visibility(source, [Finding("A", 1, 1)])
        self.assertFalse(preamble["extracted_at_all"])


class SmartBugsTests(unittest.TestCase):
    def test_classify_excludes_unsafe_negatives(self):
        with TemporaryDirectory() as root:
            base = Path(root)
            entries = []
            for folder, name, category in (
                ("reentrancy", "dao.sol", "reentrancy"),
                ("unchecked_low_level_calls", "dao.sol", "unchecked_low_level_calls"),
                ("unchecked_low_level_calls", "keep.sol", "unchecked_low_level_calls"),
                ("arithmetic", "math.sol", "arithmetic"),
            ):
                path = base / "dataset" / folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("contract A {}", encoding="utf-8")
                entries.append({"path": f"dataset/{folder}/{name}", "vulnerabilities": [{"category": category}]})
            import json

            (base / "vulnerabilities.json").write_text(json.dumps(entries), encoding="utf-8")
            positive, negative = classify(base)
            self.assertEqual([p.name for p in positive], ["dao.sol"])
            self.assertEqual([p.name for p in negative], ["math.sol"])


if __name__ == "__main__":
    unittest.main()
