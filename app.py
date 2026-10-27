"""Flask service (deployed on Heroku) that shells out to the C++ engine.
POST /predict  {"profile": [256 floats]}  ->  {"counterfeit_probability": p, "label": "counterfeit"|"genuine"}
"""
import os, subprocess
from flask import Flask, jsonify, request

ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(ROOT, "bin", "engine")
WEIGHTS = os.environ.get("WEIGHTS", os.path.join(ROOT, "artifacts", "weights.txt"))
PROFILE_LEN = 256
app = Flask(__name__)


def run_engine(rows):
    inp = "\n".join(",".join(repr(float(v)) for v in r) for r in rows) + "\n"
    res = subprocess.run([ENGINE, WEIGHTS], input=inp, capture_output=True, text=True, timeout=10)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "engine failed")
    return [float(x) for x in res.stdout.split()]


@app.get("/health")
def health():
    return jsonify(status="ok", engine=os.path.exists(ENGINE), weights=os.path.exists(WEIGHTS))


@app.post("/predict")
def predict():
    body = request.get_json(silent=True) or {}
    prof = body.get("profile")
    if not isinstance(prof, list) or len(prof) != PROFILE_LEN or not all(isinstance(v, (int, float)) for v in prof):
        return jsonify(error=f"'profile' must be a list of {PROFILE_LEN} numbers"), 400
    p = run_engine([prof])[0]
    return jsonify(counterfeit_probability=round(p, 6), label="counterfeit" if p > 0.5 else "genuine")


@app.post("/predict/batch")
def predict_batch():
    rows = (request.get_json(silent=True) or {}).get("profiles")
    if not isinstance(rows, list) or not rows or len(rows) > 500 or any(len(r) != PROFILE_LEN for r in rows):
        return jsonify(error=f"'profiles' must be 1-500 lists of {PROFILE_LEN} numbers"), 400
    return jsonify(probabilities=[round(p, 6) for p in run_engine(rows)])


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)))
