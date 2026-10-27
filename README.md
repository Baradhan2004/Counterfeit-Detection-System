# Counterfeit Banknote Detection (C++ · RNN · Heroku)

A recurrent neural network, **implemented from scratch** (NumPy, manual back-propagation through time, Adam), that reads a banknote scan-line as a *sequence* of 32 windows and decides genuine vs counterfeit.
Inference runs in a **C++17 engine** (feature extraction + RNN forward pass) behind a Flask API deployed on **Heroku**.

```
scan-line (256 samples) ─> 32 windows × 4 features ─> Elman RNN (H=16) ─> sigmoid ─> P(counterfeit)
  mean · std · mean|Δ| · zero-crossings      353 parameters
python/ trains + exports weights.txt  ──>  cpp/engine.cpp loads them  ──>  app.py (gunicorn) on Heroku
```

## Run
```bash
make                                   # builds bin/engine (g++ -O2 -std=c++17, 0 warnings)
python3 python/train.py                # synthetic scan-lines; or: --csv data_banknote_authentication.txt  (UCI)
python3 tests/test_all.py              # 6 tests incl. gradient check + C++/NumPy parity
./bin/engine artifacts/weights.txt --bench 20000 < profiles.csv
python3 app.py                         # POST /predict {"profile":[256 floats]}
```
Heroku: `Procfile` (gunicorn) + `bin/post_compile` (compiles the C++ engine on the dyno) + `runtime.txt`. Commit `artifacts/weights.txt` before deploying.

## Measured results
**Correctness of the implementation**
- BPTT gradients vs central finite differences: **max relative error 5.4e-8** over every parameter.
- C++ engine vs NumPy on 300 held-out profiles: **max |Δprobability| = 2.2e-15**.
- Latency (feature extraction + RNN): **C++ ≈ 34.9 µs (≈28,700 inf/s)** vs NumPy single-sample ≈ 338 µs (≈2,950 inf/s) → **≈9.7× faster**.

**Model quality on the bundled *synthetic* generator** (6,000 train / 1,500 val / 1,500 test, balanced; this is a pipeline check, not a real-world claim)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Elman RNN (353 params) | 92.5% | 96.1% | 88.7% | 92.2% | 0.958 |
| Logistic regression on pooled features | 92.3% | – | – | 91.9% | 0.968 |

On this synthetic data the RNN does **not** beat the pooled-feature baseline (reported honestly). To get real numbers, train on the UCI Banknote Authentication dataset (`--csv`, 1,372 notes) and record them below.

| Real-data result (fill in) | |
|---|---|
| Dataset / split | UCI Banknote Authentication, 70/15/15 |
| Accuracy / F1 / AUC | _run `python3 python/train.py --csv ...`_ |

## Layout
`python/` (features, synthetic data, RNN, training) · `cpp/engine.cpp` · `app.py` · `tests/` · `docs/` (raw logs)
