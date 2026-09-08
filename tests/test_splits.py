import unittest

from config.config import ExperimentConfig
from data import load_records
from data.loader import load_records
from experiments.splits import make_outer_splits, make_validation_split


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


if __name__ == "__main__":
    unittest.main()
