import itertools
import unittest

import numpy as np

from server.services.gop_service import GOPEvaluator


def exhaustive_best_path(log_probs: np.ndarray, labels: list[int], blank: int) -> list[int]:
    """Reference enumeration for the small trellises used in these tests."""

    state_tokens = [blank]
    for label in labels:
        state_tokens.extend([label, blank])
    final_states = {len(state_tokens) - 2, len(state_tokens) - 1}
    best_score = -np.inf
    best_path = None

    for path in itertools.product(range(len(state_tokens)), repeat=len(log_probs)):
        if path[0] not in {0, 1} or path[-1] not in final_states:
            continue
        valid = True
        for previous, current in zip(path, path[1:]):
            delta = current - previous
            if delta not in {0, 1, 2}:
                valid = False
                break
            if delta == 2:
                if current % 2 == 0 or state_tokens[current] == state_tokens[current - 2]:
                    valid = False
                    break
        if not valid:
            continue
        score = sum(log_probs[t, state_tokens[state]] for t, state in enumerate(path))
        if score > best_score:
            best_score = score
            best_path = list(path)

    if best_path is None:
        raise AssertionError("reference enumeration found no valid CTC path")
    return best_path


class CTCViterbiTests(unittest.TestCase):
    def evaluator(self) -> GOPEvaluator:
        evaluator = GOPEvaluator.__new__(GOPEvaluator)
        evaluator.blank_id = 0
        return evaluator

    def test_path_matches_exhaustive_ctc_search(self):
        log_probs = np.log(
            np.array(
                [
                    [0.60, 0.30, 0.10],
                    [0.15, 0.75, 0.10],
                    [0.70, 0.10, 0.20],
                    [0.10, 0.15, 0.75],
                    [0.70, 0.10, 0.20],
                ],
                dtype=np.float64,
            )
        )
        labels = [1, 2]
        _, path = self.evaluator().viterbi_ctc_align(log_probs, labels)
        self.assertEqual(path.tolist(), exhaustive_best_path(log_probs, labels, 0))

    def test_repeated_phones_require_an_intervening_blank(self):
        log_probs = np.log(
            np.array(
                [
                    [0.10, 0.90],
                    [0.80, 0.20],
                    [0.10, 0.90],
                    [0.80, 0.20],
                ],
                dtype=np.float64,
            )
        )
        unit_frames, path = self.evaluator().viterbi_ctc_align(log_probs, [1, 1])
        self.assertIn(2, path.tolist())
        self.assertTrue(unit_frames[0])
        self.assertTrue(unit_frames[1])
        self.assertLess(max(unit_frames[0]), min(unit_frames[1]))


if __name__ == "__main__":
    unittest.main()
