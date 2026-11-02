"""M3 definition-of-done checks. Exit 0 means ready to tag v0.3.0-M3.
Needs: `make train-fraud` and `make typologies` run, MLFLOW_TRACKING_URI exported.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    import dotenv
    dotenv.load_dotenv(ROOT / ".env")
except ImportError:
    pass

FILES = [
    "ml/src/aegis_ml/training/splits.py",
    "ml/src/aegis_ml/training/prepare_ieee.py",
    "ml/src/aegis_ml/training/matrix.py",
    "ml/src/aegis_ml/training/train_fraud.py",
    "ml/src/aegis_ml/training/registry.py",
    "ml/src/aegis_ml/models/fraud_models.py",
    "ml/src/aegis_ml/models/calibration.py",
    "ml/src/aegis_ml/evaluation/gates.py",
    "ml/src/aegis_ml/evaluation/metrics.py",
    "ml/src/aegis_ml/evaluation/rules_baseline.py",
    "ml/src/aegis_ml/evaluation/fairness.py",
    "ml/src/aegis_ml/evaluation/typology_eval.py",
    "ml/src/aegis_ml/evaluation/reports.py",
    "scripts/champion.py",
    "docs/adr/0005-m3-fraud-models.md",
    "docs/model_card_fraud.md",
    ".github/workflows/retrain.yml",
    "artifacts/fraud/metrics.json",
    "reports/m3_fraud.json",
]
results = []


def ok(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'OK ' if cond else 'XX '}] {name} {detail}")


def run(args):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(["ml/src", "."])}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env).returncode


for f in FILES:
    ok(f"file {f}", (ROOT / f).exists())

ok("unit tests", run(["-m", "pytest", "-q", "tests/m3/unit"]) == 0)

path = ROOT / "reports/m3_fraud.json"
report = json.loads(path.read_text()) if path.exists() else {"gates": []}
ok("all fraud gates pass", bool(report["gates"]) and all(g["pass"] for g in report["gates"]),
   f"({sum(g['pass'] for g in report['gates'])}/{len(report['gates'])})")

meta_path = ROOT / "artifacts/fraud/metrics.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    champ = meta["results"][meta["champion"]]
    ok("recall@1%FPR beats the rules baseline by >= 30% (objective 5.2)",
       champ["lift_over_rules"] >= 0.30, f"({champ['lift_over_rules']:.1%})")
    ok("calibrated within 5 points (ECE)", champ["ece"] <= 0.05, f"({champ['ece']:.4f})")
    ok("both candidate algorithms were trained and compared",
       len(meta["results"]) >= 2, f"({', '.join(meta['results'])})")
    ok("frequency encoding persisted with the model",
       bool(meta.get("frequency_encoding")))
    ok("feature order persisted with the model", bool(meta.get("feature_names")))
    ok("disparate-impact screen recorded", "disparate_impact" in meta)
else:
    ok("artifacts/fraud/metrics.json", False)

typ = ROOT / "artifacts/fraud/typologies.json"
if typ.exists():
    t = json.loads(typ.read_text())
    ok("typology detection >= 80% (objective 5.4)", t["overall"]["rate"] >= 0.80,
       f"({t['overall']['rate']:.1%})")
else:
    ok("artifacts/fraud/typologies.json - run `make typologies`", False)

if not os.getenv("MLFLOW_TRACKING_URI"):
    ok("MLFLOW_TRACKING_URI exported", False)
else:
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        for alias in ("champion", "challenger"):
            v = client.get_model_version_by_alias("aegisflow-fraud", alias)
            ok(f"alias '{alias}' set", v is not None, f"(v{v.version})" if v else "")
        ok("calibrator logged alongside the model",
           any(f.path == "calibrator" for f in client.list_artifacts(
               client.get_model_version_by_alias("aegisflow-fraud", "champion").run_id)))
    except Exception as exc:  # noqa: BLE001
        ok("MLflow registry reachable with both aliases", False, f"({exc})")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
