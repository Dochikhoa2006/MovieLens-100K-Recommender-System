from contextlib import redirect_stdout
from io import StringIO
import re
import unittest

import numpy as np
from scipy import sparse

from Factorization_Machine_Library import FM_Regression


class FactorizationMachineTests(unittest.TestCase):
    def test_predict_matches_pairwise_scores_for_csr_and_csc(self):
        model = FM_Regression(rank=2)
        model.w0 = 3.0
        model.w = np.array([0.5, -0.25, 0.0])
        model.V = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, -0.1]])
        rows = sparse.csr_matrix([[1, 1, 0], [2, 0, 1], [0, 0, 0]], dtype=float)
        expected = np.array([3.36, 4.06, 3.0])

        for matrix in (rows, rows.tocsc()):
            np.testing.assert_allclose(model.predict(matrix), expected)

    def test_seeded_training_is_repeatable(self):
        rows = sparse.csr_matrix([[1, 0, 2], [0, 1, 1], [1, 1, 0]], dtype=float)
        ratings = np.array([4.0, 2.0, 3.0])
        models = []

        for _ in range(2):
            model = FM_Regression(n_iter=2, rank=3, step_size=0.01, random_state=150)
            with redirect_stdout(StringIO()):
                model.fit(rows, ratings)
            models.append(model)

        np.testing.assert_array_equal(models[0].w, models[1].w)
        np.testing.assert_array_equal(models[0].V, models[1].V)

    def test_training_rmse_uses_unclipped_residual(self):
        rows = sparse.csr_matrix([[1e6, 1e6], [0, 0]], dtype=float)
        model = FM_Regression(n_iter=1, rank=1, step_size=0, random_state=42)
        output = StringIO()

        with redirect_stdout(output):
            model.fit(rows, np.array([0.0, 0.0]))

        reported = float(re.search(r"RMSE: ([\d.]+)", output.getvalue()).group(1))
        expected = abs(1e12 * model.V[0, 0] * model.V[1, 0]) / np.sqrt(2)
        self.assertAlmostEqual(reported, expected, delta=0.0001)
        self.assertIn("Divergence detected at Epoch 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
