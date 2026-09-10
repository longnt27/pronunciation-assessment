import itertools
import unittest

import numpy as np
import torch

from benchmarks.compare_three_systems import (
    cao_alignment_free_features,
    ctc_nlls,
    sequence_diagonal_pairs,
    viterbi_regions,
)
from benchmarks.three_system_models import GOPTJoint, GOPTPhone


def collapse(path: tuple[int, ...], blank: int = 0) -> list[int]:
    result = []
    previous = None
    for token in path:
        if token != previous and token != blank:
            result.append(token)
        previous = token
    return result


def brute_ctc_nll(log_probs: torch.Tensor, target: list[int]) -> float:
    probability = 0.0
    for path in itertools.product(range(log_probs.shape[1]), repeat=len(log_probs)):
        if collapse(path) == target:
            probability += float(torch.exp(sum(log_probs[t, token] for t, token in enumerate(path))))
    return -float(np.log(probability))


class CaoFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        probabilities = torch.tensor([
            [0.55, 0.35, 0.10],
            [0.20, 0.65, 0.15],
            [0.25, 0.20, 0.55],
            [0.60, 0.10, 0.30],
        ], dtype=torch.float64)
        self.log_probs = probabilities.log()

    def test_ctc_loss_matches_exhaustive_path_sum(self) -> None:
        targets = [[1, 2], [1, 1], [2], []]
        actual = ctc_nlls(self.log_probs, targets).numpy()
        expected = np.asarray([
            brute_ctc_nll(self.log_probs, target) for target in targets
        ])
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_alignment_free_vector_order_and_values(self) -> None:
        features = cao_alignment_free_features(
            self.log_probs.float(), [1, 2], [1, 2]
        )
        self.assertEqual(features.shape, (2, 4))
        canonical = float(ctc_nlls(self.log_probs, [[1, 2]])[0])
        alternatives = [
            [[2], [1, 2], [2, 2]],
            [[1], [1, 1], [1, 2]],
        ]
        expected = []
        for phone_alternatives in alternatives:
            nll = ctc_nlls(self.log_probs, phone_alternatives).numpy()
            expected.append([canonical, *(nll - canonical)])
        np.testing.assert_allclose(features, expected, rtol=1e-5, atol=1e-5)

    def test_viterbi_regions_are_contiguous_for_repeated_phones(self) -> None:
        regions = viterbi_regions(
            self.log_probs.float(), [1, 1], ["AA", "AA"], duration=0.4
        )
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0]["start"], 0.0)
        self.assertEqual(regions[-1]["end"], 0.4)
        self.assertAlmostEqual(regions[0]["end"], regions[1]["start"])

    def test_mfa_edit_alignment_keeps_substitution_intervals(self) -> None:
        self.assertEqual(
            sequence_diagonal_pairs(
                ["K", "AO", "L", "IH", "T"],
                ["K", "AA", "L", "T"],
            ),
            [(0, 0), (1, 1), (2, 2), (4, 3)],
        )


class GOPTShapeTest(unittest.TestCase):
    def test_phone_and_joint_shapes(self) -> None:
        features = torch.randn(3, 50, 41)
        phones = torch.randint(-1, 39, (3, 50))
        self.assertEqual(GOPTPhone(41)(features, phones).shape, (3, 50))
        phone, word, utterance = GOPTJoint(41)(features, phones)
        self.assertEqual(phone.shape, (3, 50))
        self.assertEqual(word.shape, (3, 50, 3))
        self.assertEqual(utterance.shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
