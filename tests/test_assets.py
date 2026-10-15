from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assets import require_inference_assets


class AssetTests(unittest.TestCase):
    def test_missing_files_are_reported_together(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "FM.pkl").touch()

            with self.assertRaises(FileNotFoundError) as result:
                require_inference_assets(root)

        message = str(result.exception)
        self.assertIn("Factorization_Model_Trained.pkl", message)
        self.assertIn("MovieLens_100k_Dataset/ratings.csv", message)
        self.assertNotIn("FM.pkl", message)
        self.assertIn("README.md", message)


if __name__ == "__main__":
    unittest.main()
