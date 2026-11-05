import unittest

from config.config import ExperimentConfig
from data.loader import load_records
from experiments.splits import make_outer_splits, make_validation_split
from representation.pruning import deduplicate_groups


class GroupSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records("Dataset/reentrancy_1671.txt")
        cls.config = ExperimentConfig()

    def test_every_outer_fold_is_group_disjoint(self):
        for train_indices, test_indices in make_outer_splits(self.records, self.config):
            train_groups = {self.records[index].group_id for index in train_indices}
            test_groups = {self.records[index].group_id for index in test_indices}
            self.assertFalse(train_groups & test_groups)

    def test_validation_is_group_disjoint_and_has_both_classes(self):
        train_indices, _ = make_outer_splits(self.records, self.config)[0]
        outer_train = [self.records[index] for index in train_indices]
        fit_indices, validation_indices = make_validation_split(outer_train, self.config)
        fit = [outer_train[index] for index in fit_indices]
        validation = [outer_train[index] for index in validation_indices]
        self.assertFalse(
            {record.group_id for record in fit}
            & {record.group_id for record in validation}
        )
        self.assertEqual({record.label for record in fit}, {0, 1})
        self.assertEqual({record.label for record in validation}, {0, 1})

    def test_dedup_five_fold_cv_is_paired_across_extraction_strategies(self):
        first_dedup = deduplicate_groups(
            load_records("Dataset/reentrancy_1671.txt", extraction_strategy="first")
        )
        flow_dedup = deduplicate_groups(
            load_records("Dataset/reentrancy_1671.txt", extraction_strategy="call_flow")
        )
        self.assertEqual(len(first_dedup), 185)
        self.assertEqual(len(flow_dedup), 185)
        first_splits = make_outer_splits(first_dedup, self.config)
        flow_splits = make_outer_splits(flow_dedup, self.config)
        for (tr1, te1), (tr2, te2) in zip(first_splits, flow_splits):
            self.assertEqual(list(tr1), list(tr2))
            self.assertEqual(list(te1), list(te2))


if __name__ == "__main__":
    unittest.main()
