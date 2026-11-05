import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from data.extraction import split_call_anchored_regions, split_call_flow_regions
from data.loader import load_records
from data.types import GadgetRecord
from data.parser import parse_gadgets
from representation.sequences import (
    C_SEGMENT,
    PAD_SEGMENT,
    SEP_SEGMENT,
    build_sequence,
    coverage,
)
from representation.variants import VARIANTS


class SequencePolicyTests(unittest.TestCase):
    def setUp(self):
        self.record = GadgetRecord(
            sample_id="sample",
            label=1,
            w_tokens=tuple(f"w{i}" for i in range(10)),
            c_tokens=tuple(f"c{i}" for i in range(10)),
            group_id="group",
        )

    def test_prefix_budget_always_retains_separator_and_c(self):
        item = build_sequence(self.record, VARIANTS["b3"], max_len=10, w_ratio=0.6)
        self.assertEqual(
            item.tokens,
            ("w0", "w1", "w2", "w3", "w4", "[SEP]", "c0", "c1", "c2", "c3"),
        )
        self.assertEqual(item.segment_ids[5], SEP_SEGMENT)
        np.testing.assert_array_equal(item.segment_ids[6:], [C_SEGMENT] * 4)

    def test_head_tail_uses_both_ends(self):
        item = build_sequence(self.record, VARIANTS["b4"], max_len=9, w_ratio=0.5)
        self.assertEqual(item.tokens[:4], ("w0", "w1", "w8", "w9"))
        self.assertEqual(item.tokens[-4:], ("c0", "c1", "c8", "c9"))

    def test_empty_c_still_has_separator_and_padding(self):
        record = GadgetRecord("sample", 0, ("w",), (), "group")
        item = build_sequence(record, VARIANTS["b3"], max_len=5, w_ratio=0.5)
        self.assertEqual(item.tokens, ("w", "[SEP]"))
        np.testing.assert_array_equal(
            item.segment_ids,
            [0, SEP_SEGMENT, PAD_SEGMENT, PAD_SEGMENT, PAD_SEGMENT],
        )

    def test_coverage(self):
        item = build_sequence(self.record, VARIANTS["b3"], max_len=10, w_ratio=0.6)
        metrics = coverage([self.record], [item])
        self.assertEqual(metrics["sep_coverage"], 1.0)
        self.assertEqual(metrics["c_coverage"], 1.0)
        self.assertAlmostEqual(metrics["w_retention"], 0.5)
        self.assertAlmostEqual(metrics["c_retention"], 0.4)


class DatasetIntegrationTests(unittest.TestCase):
    def test_repository_dataset(self):
        records = load_records("Dataset/reentrancy_1671.txt")
        self.assertEqual(len(records), 1671)
        self.assertEqual(sum(record.label for record in records), 576)
        self.assertTrue(all(record.w_tokens for record in records))
        self.assertGreater(sum(bool(record.c_tokens) for record in records), 1600)
        self.assertEqual(len({record.sample_id for record in records}), 1671)
        self.assertEqual(len({record.group_id for record in records}), 185)
        self.assertEqual(
            sum(record.diagnostics.w_contains_modifier_declaration for record in records),
            123,
        )
        self.assertEqual(
            sum(record.diagnostics.w_contains_contract_declaration for record in records),
            102,
        )

    def test_parser_rejects_a_missing_label(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text("1 sample.sol\nfunction f() {\n}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing binary label"):
                list(parse_gadgets(path))

    def test_parser_keeps_the_final_standalone_binary_value_as_label(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "numeric-code.txt"
            path.write_text(
                "1 sample.sol\nfunction f() {\n1\n}\n0\n" + "-" * 33 + "\n",
                encoding="utf-8",
            )
            parsed = list(parse_gadgets(path))
        self.assertEqual(parsed[0].label, 0)
        self.assertNotIn("1", parsed[0].lines)


class AblationSequenceTests(unittest.TestCase):
    def setUp(self):
        self.record = GadgetRecord("sample", 1, ("w0", "w1"), ("c0", "c1"), "g")

    def test_b0_and_b1_use_the_same_flat_sequence(self):
        b0 = build_sequence(self.record, VARIANTS["b0"], 4, 0.5)
        b1 = build_sequence(self.record, VARIANTS["b1"], 4, 0.5)
        self.assertEqual(b0.tokens, ("w0", "w1", "c0", "c1"))
        self.assertEqual(b0.tokens, b1.tokens)

    def test_b2_adds_separator_without_requiring_segment_embedding(self):
        item = build_sequence(self.record, VARIANTS["b2"], 5, 0.5)
        self.assertEqual(item.tokens, ("w0", "w1", "[SEP]", "c0", "c1"))
        self.assertFalse(VARIANTS["b2"].use_segment_embedding)


class CallExtractionStrategyTests(unittest.TestCase):
    def setUp(self):
        self.contract_lines = [
            "pragma solidity ^0.8.0;",
            "contract Vault {",
            "mapping(address => uint256) public balances;",
            "function getBalance(address user) public view returns (uint256) {",
            "return balances[user];",
            "}",
            "function _sendEth(address to, uint256 amount) internal {",
            "(bool ok, ) = to.call.value(amount)(\"\");",
            "require(ok);",
            "}",
            "function withdraw(uint256 amount) external {",
            "require(balances[msg.sender] >= amount);",
            "_sendEth(msg.sender, amount);",
            "balances[msg.sender] -= amount;",
            "}",
            "function setFee(uint256 fee) external {",
            "}",
            "}",
        ]

    def test_call_anchored_skips_Leading_view_getter_and_selects_withdraw(self):
        w_lines, c_lines, _ = split_call_anchored_regions(self.contract_lines, w_budget=60)
        self.assertTrue(w_lines[0].startswith("function withdraw"))
        self.assertTrue(any("getBalance" in line for line in c_lines))

    def test_call_flow_prioritizes_internal_callee_and_state_in_c(self):
        w_lines, c_lines, _ = split_call_flow_regions(
            self.contract_lines, w_budget=25, c_budget=200
        )
        self.assertTrue(w_lines[0].startswith("function withdraw"))
        # _sendEth is called by withdraw and makes the low-level call, so it must lead C
        self.assertTrue(c_lines[0].startswith("function _sendEth"))
        self.assertTrue(any("balances" in line for line in c_lines))

    def test_group_id_is_invariant_across_extraction_strategies(self):
        first_recs = load_records("Dataset/reentrancy_1671.txt", extraction_strategy="first")
        flow_recs = load_records("Dataset/reentrancy_1671.txt", extraction_strategy="call_flow")
        self.assertEqual(
            [r.group_id for r in first_recs],
            [r.group_id for r in flow_recs],
        )


if __name__ == "__main__":
    unittest.main()
