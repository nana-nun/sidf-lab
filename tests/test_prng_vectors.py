import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VECTOR_PATH = ROOT / "references" / "prng-test-vectors" / "philox4x32-10.json"
MASK32 = 0xFFFFFFFF
M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85


def _u32(value):
    return value & MASK32


def _mulhilo(multiplier, value):
    product = multiplier * value
    return (product >> 32) & MASK32, product & MASK32


def _philox4x32_round(counter, key):
    hi0, lo0 = _mulhilo(M0, counter[0])
    hi1, lo1 = _mulhilo(M1, counter[2])
    return [
        _u32(hi1 ^ counter[1] ^ key[0]),
        lo1,
        _u32(hi0 ^ counter[3] ^ key[1]),
        lo0,
    ]


def _philox4x32_10(counter, key):
    counter = list(counter)
    key = list(key)
    for _ in range(10):
        counter = _philox4x32_round(counter, key)
        key = [_u32(key[0] + W0), _u32(key[1] + W1)]
    return counter


def _parse_words(words):
    return [int(word, 16) for word in words]


class TestPhiloxVectors(unittest.TestCase):
    def test_saved_vectors_match_reference_round_function(self):
        with VECTOR_PATH.open(encoding="utf-8") as fh:
            vector = json.load(fh)

        self.assertEqual(vector["variant"], "Philox4x32")
        self.assertEqual(vector["rounds"], 10)
        self.assertEqual(vector["word_width_bits"], 32)
        self.assertEqual(vector["counter_layout"], ["stage", "sweep", "pixel_index", "purpose"])

        for case in vector["cases"]:
            with self.subTest(case=case["name"]):
                key = _parse_words(case["key_words_hex"])
                counter = _parse_words(case["counter_words_hex"])
                expected = _parse_words(case["output_words_hex"])
                self.assertEqual(_philox4x32_10(counter, key), expected)


if __name__ == "__main__":
    unittest.main()
