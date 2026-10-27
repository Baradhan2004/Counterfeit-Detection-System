import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import window_features
from rnn import ElmanRNN
from synthetic import make_dataset


def metrics(y, p):
    pred = (p > 0.5).astype(int); y = y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    # ROC-AUC via rank statistic
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return dict(accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12), auc=float(auc),
                confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn))


def load_uci_csv(path):
    """UCI Banknote Authentication CSV (variance, skewness, kurtosis, entropy, class).
    The 4 statistics are fed as a length-4 sequence of 1 feature each."""
    a = np.loadtxt(path, delimiter=",")
    return a[:, :4, None], a[:, 4]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="UCI banknote CSV; omit to use the synthetic scan-line generator")
    ap.add_argument("--out", default="artifacts"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=16)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)

    if a.csv:
        X, y = load_uci_csv(a.csv); mode = "raw"
        idx = np.random.default_rng(0).permutation(len(X)); X, y = X[idx], y[idx]
        n = len(X); tr, va = slice(0, int(.7 * n)), slice(int(.7 * n), int(.85 * n)); te = slice(int(.85 * n), n)
        Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    else:
        mode = "window_features"
        (rtr, ytr), (rva, yva), (rte, yte) = make_dataset(6000, 1), make_dataset(1500, 2), make_dataset(1500, 3)
        Xtr, Xva, Xte = window_features(rtr), window_features(rva), window_features(rte)
        np.save(out / "raw_test_profiles.npy", rte); np.save(out / "test_labels.npy", yte)

    mu = Xtr.reshape(-1, Xtr.shape[2]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-8
    norm = lambda z: (z - mu) / sd
    model = ElmanRNN(Xtr.shape[2], a.hidden, seed=0)
    print(f"mode={mode} train={len(Xtr)} val={len(Xva)} test={len(Xte)} T={Xtr.shape[1]} F={Xtr.shape[2]} params={model.n_params}")
    t = time.time(); model.fit(norm(Xtr), ytr, norm(Xva), yva, epochs=a.epochs); train_s = time.time() - t
    ptest = model.predict_proba(norm(Xte)); m = metrics(yte, ptest); m["train_seconds"] = round(train_s, 1); m["params"] = model.n_params
    print("TEST:", json.dumps(m, indent=2))

    # baseline: logistic regression on mean-pooled window features
    from sklearn.linear_model import LogisticRegression
    pool = lambda z: np.concatenate([z.mean(1), z.max(1)], axis=1)
    lr = LogisticRegression(max_iter=2000).fit(pool(norm(Xtr)), ytr)
    mb = metrics(yte, lr.predict_proba(pool(norm(Xte)))[:, 1])
    print(f"BASELINE logistic regression (pooled features): acc={mb['accuracy']:.4f} f1={mb['f1']:.4f} auc={mb['auc']:.4f}")
    m["baseline_logreg"] = mb
    model.export_text(out / "weights.txt", mu, sd, mode)
    np.save(out / "test_probs_python.npy", ptest)
    json.dump(m, open(out / "metrics.json", "w"), indent=2)
