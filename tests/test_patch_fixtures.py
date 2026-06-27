import unittest

import numpy as np

from sidf_lab.patch_fixtures import (
    load_grayscale_patch,
    load_patch_manifest,
    list_patch_records,
    validate_patch_manifest,
)


class PatchFixtureTests(unittest.TestCase):
    def test_manifest_separates_sources_by_split(self) -> None:
        manifest = load_patch_manifest()
        source_splits = {source["source_id"]: source["split"] for source in manifest["sources"]}
        self.assertIn("development", set(source_splits.values()))
        self.assertIn("evaluation", set(source_splits.values()))

        for patch in manifest["patches"]:
            self.assertEqual(patch["split"], source_splits[patch["source_id"]])

    def test_split_listing_and_patch_loading(self) -> None:
        development = list_patch_records("development")
        evaluation = list_patch_records("evaluation")
        self.assertEqual([patch["name"] for patch in development], ["dev_hobbema_landscape"])
        self.assertEqual([patch["name"] for patch in evaluation], ["eval_hokusai_wave"])

        for patch in development + evaluation:
            image = load_grayscale_patch(patch["name"])
            self.assertEqual(image.shape, (128, 128))
            self.assertEqual(image.dtype, np.float64)
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)

    def test_manifest_rejects_patch_split_mismatch(self) -> None:
        manifest = load_patch_manifest()
        broken = {
            **manifest,
            "patches": [
                {
                    **manifest["patches"][0],
                    "split": "evaluation",
                }
            ],
        }
        with self.assertRaises(ValueError):
            validate_patch_manifest(broken)

    def test_unknown_patch_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            load_grayscale_patch("missing_patch")


if __name__ == "__main__":
    unittest.main()
