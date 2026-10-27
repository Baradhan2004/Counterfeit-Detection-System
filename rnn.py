"""Elman RNN binary classifier implemented from scratch in NumPy (manual BPTT, Adam, grad clipping)."""
import numpy as np


class ElmanRNN:
    def __init__(self, n_in, n_hidden=16, seed=0):
        r = np.random.default_rng(seed)
        self.F, self.H = n_in, n_hidden
        self.p = {
            "Wxh": r.normal(0, np.sqrt(1.0 / n_in), (n_in, n_hidden)),
            "Whh": r.normal(0, np.sqrt(0.5 / n_hidden), (n_hidden, n_hidden)),
            "bh": np.zeros(n_hidden),
            "Why": r.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, 1)),
            "by": np.zeros(1),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step_count = 0

    @property
    def n_params(self):
        return sum(v.size for v in self.p.values())

    def forward(self, X):
        B, T, _ = X.shape
        hs = [np.zeros((B, self.H))]
        for t in range(T):
            hs.append(np.tanh(X[:, t] @ self.p["Wxh"] + hs[-1] @ self.p["Whh"] + self.p["bh"]))
        logit = hs[-1] @ self.p["Why"] + self.p["by"]
        return 1.0 / (1.0 + np.exp(-logit)), hs

    def predict_proba(self, X):
        return self.forward(X)[0][:, 0]

    def loss_and_grads(self, X, y):
        B, T, _ = X.shape
        prob, hs = self.forward(X)
        y = y.reshape(-1, 1)
        eps = 1e-12
        loss = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        dlogit = (prob - y) / B
        g["Why"] = hs[-1].T @ dlogit
        g["by"] = dlogit.sum(0)
        dh = dlogit @ self.p["Why"].T
        for t in range(T - 1, -1, -1):          # backpropagation through time
            dz = dh * (1.0 - hs[t + 1] ** 2)
            g["Wxh"] += X[:, t].T @ dz
            g["Whh"] += hs[t].T @ dz
            g["bh"] += dz.sum(0)
            dh = dz @ self.p["Whh"].T
        return loss, g

    def adam_step(self, g, lr=3e-3, clip=5.0, b1=0.9, b2=0.999):
        norm = np.sqrt(sum((x ** 2).sum() for x in g.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.step_count += 1
        for k in self.p:
            gk = g[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk ** 2
            mh = self.m[k] / (1 - b1 ** self.step_count)
            vh = self.v[k] / (1 - b2 ** self.step_count)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return norm

    def fit(self, X, y, Xv, yv, epochs=40, batch=64, lr=3e-3, seed=0, log=print):
        r = np.random.default_rng(seed)
        best, best_p = (-1, None), None
        for ep in range(1, epochs + 1):
            idx = r.permutation(len(X)); tot = 0.0
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]
                loss, g = self.loss_and_grads(X[b], y[b]); self.adam_step(g, lr)
                tot += loss * len(b)
            acc = ((self.predict_proba(Xv) > 0.5) == yv).mean()
            if acc > best[0]:
                best = (acc, ep); best_p = {k: v.copy() for k, v in self.p.items()}
            if ep == 1 or ep % 5 == 0:
                log(f"epoch {ep:3d}  train_loss {tot/len(X):.4f}  val_acc {acc:.4f}")
        self.p = best_p
        log(f"best val_acc {best[0]:.4f} at epoch {best[1]}")
        return best

    def export_text(self, path, mu, sd, mode="window_features"):
        """Plain-text weight file consumed by cpp/engine.cpp (full float64 precision)."""
        def row(a): return " ".join(f"{x:.17g}" for x in np.ravel(a))
        with open(path, "w") as f:
            f.write(f"{mode}\n{self.F} {self.H}\n{row(mu)}\n{row(sd)}\n")
            for k in ["Wxh", "Whh", "bh", "Why", "by"]:
                f.write(row(self.p[k]) + "\n")
