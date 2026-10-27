import subprocess, sys, unittest, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT))
from features import window_features
from rnn import ElmanRNN
from synthetic import make_dataset

ENGINE, WEIGHTS = ROOT / "bin" / "engine", ROOT / "artifacts" / "weights.txt"


class TestRNN(unittest.TestCase):
    def test_gradient_check(self):
        """Analytic BPTT gradients vs central finite differences on every parameter."""
        r = np.random.default_rng(0); X = r.normal(size=(5, 6, 3)); y = r.integers(0, 2, 5).astype(float)
        m = ElmanRNN(3, 4, seed=1); _, g = m.loss_and_grads(X, y); worst = 0.0; checked = 0
        for k, v in m.p.items():
            for idx in np.ndindex(v.shape):
                old = v[idx]; v[idx] = old + 1e-6; lp, _ = m.loss_and_grads(X, y)
                v[idx] = old - 1e-6; lm, _ = m.loss_and_grads(X, y); v[idx] = old
                num = (lp - lm) / 2e-6; checked += 1
                worst = max(worst, abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx])))
        print(f"\n  gradient check: {checked} parameters, max relative error {worst:.2e}")
        self.assertLess(worst, 1e-6)

    def test_param_count(self):
        self.assertEqual(ElmanRNN(4, 16).n_params, 4 * 16 + 16 * 16 + 16 + 16 + 1)

    def test_can_overfit_tiny_batch(self):
        r = np.random.default_rng(1); X = r.normal(size=(16, 5, 2)); y = (X[:, :, 0].sum(1) > 0).astype(float)
        m = ElmanRNN(2, 8, seed=2)
        for _ in range(300):
            _, g = m.loss_and_grads(X, y); m.adam_step(g, lr=1e-2)
        self.assertGreater(((m.predict_proba(X) > .5) == y).mean(), 0.95)


class TestFeatures(unittest.TestCase):
    def test_shape_and_values(self):
        f = window_features(np.tile([0, 1, 0, 1, 0, 1, 0, 1.0], 4))
        self.assertEqual(f.shape, (1, 4, 4)); self.assertAlmostEqual(f[0, 0, 0], 0.5)
        self.assertAlmostEqual(f[0, 0, 1], 0.5); self.assertAlmostEqual(f[0, 0, 2], 1.0); self.assertEqual(f[0, 0, 3], 7)


@unittest.skipUnless(ENGINE.exists() and WEIGHTS.exists(), "run `make` and python/train.py first")
class TestCppParity(unittest.TestCase):
    def test_cpp_matches_numpy(self):
        raw = np.load(ROOT / "artifacts" / "raw_test_profiles.npy")[:300]
        py = np.load(ROOT / "artifacts" / "test_probs_python.npy")[:300]
        inp = "\n".join(",".join(f"{v:.17g}" for v in row) for row in raw) + "\n"
        out = subprocess.run([str(ENGINE), str(WEIGHTS)], input=inp, capture_output=True, text=True, check=True).stdout.split()
        diff = np.abs(np.array(out, float) - py).max()
        print(f"\n  C++ vs NumPy parity on 300 profiles: max |diff| = {diff:.2e}")
        self.assertLess(diff, 1e-9)

    def test_api(self):
        from app import app
        c = app.test_client(); self.assertEqual(c.get("/health").get_json()["status"], "ok")
        x, y = make_dataset(40, 99)
        pred = [c.post("/predict", json={"profile": r.tolist()}).get_json()["label"] == "counterfeit" for r in x]
        self.assertGreater(np.mean(np.array(pred) == y.astype(bool)), 0.8)
        self.assertEqual(c.post("/predict", json={"profile": [1, 2]}).status_code, 400)
        self.assertEqual(c.post("/predict/batch", json={"profiles": [x[0].tolist(), x[1].tolist()]}).status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
