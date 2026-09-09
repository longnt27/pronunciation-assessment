import tempfile
import unittest
from pathlib import Path

from benchmarks.evaluate_ctc_viterbi_vs_mfa import (
    parse_phone_textgrid,
    sequence_matches,
)


class AlignmentBenchmarkTests(unittest.TestCase):
    def test_sequence_match_is_monotonic_and_exact(self):
        matches = sequence_matches(
            ["B", "AH", "N", "AE", "N", "AH"],
            ["B", "AH", "N", "N", "AE", "N", "AH", "Z"],
        )
        self.assertEqual([left for left, _ in matches], list(range(6)))
        self.assertEqual(
            ["B", "AH", "N", "AE", "N", "AH"],
            [["B", "AH", "N", "N", "AE", "N", "AH", "Z"][right] for _, right in matches],
        )
        self.assertEqual(
            [right for _, right in matches],
            sorted(right for _, right in matches),
        )

    def test_phone_textgrid_parser_drops_silence_and_stress_digits(self):
        content = '''File type = "ooTextFile"
Object class = "TextGrid"
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 0.5
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "test"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.5
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.3
            text = "AH1"
        intervals [3]:
            xmin = 0.3
            xmax = 0.5
            text = "T"
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.TextGrid"
            path.write_text(content, encoding="utf-8")
            self.assertEqual(
                parse_phone_textgrid(path),
                [("AH", 0.1, 0.3), ("T", 0.3, 0.5)],
            )


if __name__ == "__main__":
    unittest.main()
