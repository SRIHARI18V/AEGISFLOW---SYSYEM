"""Champion/challenger against a real MLflow server. Skipped without
MLFLOW_TRACKING_URI. Uses a throwaway model name so it cannot touch the real one."""
import os
import uuid
import pytest

pytestmark = pytest.mark.skipif(not os.getenv("MLFLOW_TRACKING_URI"),
                                reason="needs the MLflow server")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from aegis_ml.training import registry  # noqa: E402


@pytest.fixture(autouse=True)
def clean_mlflow_run():
    mlflow.end_run()
    yield
    mlflow.end_run()


@pytest.fixture()
def scratch_model(monkeypatch):
    name = f"test-aegis-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(registry, "MODEL", name)
    yield name
    client = MlflowClient()
    try:
        client.delete_registered_model(name)
    except Exception:  # noqa: BLE001
        pass


def _log(recall):
    mlflow.end_run()
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    with mlflow.start_run() as run:
        mlflow.log_metric("recall_at_1pct_fpr", recall)
        mlflow.log_metric("lift_over_rules", 0.35)
        mlflow.log_metric("pr_auc", 0.71)
        mlflow.log_metric("ece", 0.03)
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id
    return run_id, model


def test_first_model_becomes_both_aliases(scratch_model):
    from aegis_ml.models.calibration import Calibrator
    run_id, model = _log(0.50)
    cal = Calibrator().fit([0.1, 0.9], [0, 1])
    d = registry.register_candidate(run_id, model, cal, ["f"], {"recall_at_1pct_fpr": 0.50})
    assert d.changed and "champion and challenger" in d.reason
    client = MlflowClient()
    assert client.get_model_version_by_alias(scratch_model, "champion").version == d.version


def test_a_tiny_lift_does_not_promote(scratch_model):
    from aegis_ml.models.calibration import Calibrator
    cal = Calibrator().fit([0.1, 0.9], [0, 1])
    run_a, model_a = _log(0.50)
    registry.register_candidate(run_a, model_a, cal, ["f"], {"recall_at_1pct_fpr": 0.50})
    run_b, model_b = _log(0.505)
    registry.register_candidate(run_b, model_b, cal, ["f"], {"recall_at_1pct_fpr": 0.505})
    decision = registry.compare()
    assert not decision.changed and "noise band" in decision.reason
    assert not registry.promote_challenger("tester").changed
