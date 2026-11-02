"""Champion/challenger registry policy (FR-7).
Two named aliases, never "latest": `champion` is what the scoring service serves,
`challenger` is what it shadow-scores. A new candidate always lands as challenger.
Swapping champion and challenger is a separate, human, audited action - the
scoring service picks the change up on its next registry poll.
"""
from __future__ import annotations

from dataclasses import dataclass
import mlflow
from mlflow.tracking import MlflowClient

from ..evaluation.gates import challenger_wins

MODEL = "aegisflow-fraud"
CHAMPION, CHALLENGER = "champion", "challenger"


@dataclass
class Decision:
    changed: bool
    reason: str
    version: str | None = None


def _alias_version(client: MlflowClient, alias: str):
    try:
        return client.get_model_version_by_alias(MODEL, alias)
    except Exception: # noqa: BLE001 - alias simply not set yet
        return None


def register_candidate(run_id: str, model, calibrator, feature_names,
                       metrics: dict, client: MlflowClient | None = None) -> Decision:
    """Log the model plus its calibrator and column order as one artifact, register
    it, and set the `challenger` alias. The calibrator travels WITH the model:
    shipping them separately is how a service ends up serving raw margins."""
    client = client or MlflowClient()
    mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.sklearn.log_model(calibrator.iso, artifact_path="calibrator")
    mlflow.log_dict({"feature_names": list(feature_names)}, "model/columns.json")
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL).version
    client.set_registered_model_alias(MODEL, CHALLENGER, version)
    for k, v in metrics.items():
        if isinstance(v, float):
            client.set_model_version_tag(MODEL, version, k, f"{v:.6f}")
    champ = _alias_version(client, CHAMPION)
    if champ is None:
        client.set_registered_model_alias(MODEL, CHAMPION, version)
        return Decision(True, f"first model: v{version} is champion and challenger",
                        version)
    return Decision(True, f"v{version} registered as challenger (champion stays "
                          f"v{champ.version}); run `make compare` to decide", version)


def compare(client: MlflowClient | None = None) -> Decision:
    """Read both aliases' metrics and apply the promotion rule. Prints the reason
    either way - "no change" with a reason is a result, not a failure."""
    client = client or MlflowClient()
    champ, chal = _alias_version(client, CHAMPION), _alias_version(client, CHALLENGER)
    if champ is None or chal is None:
        return Decision(False, "both champion and challenger aliases must be set")
    if champ.version == chal.version:
        return Decision(False, "champion and challenger are the same version")

    def metrics(v):
        return {k: float(val) for k, val in client.get_run(v.run_id).data.metrics.items()}

    ok, reason = challenger_wins(metrics(chal), metrics(champ))
    return Decision(ok, reason, chal.version)


def promote_challenger(actor: str, client: MlflowClient | None = None) -> Decision:
    """Human action. Re-checks the rule first, so a stale terminal cannot promote a
    model that stopped winning."""
    client = client or MlflowClient()
    decision = compare(client)
    if not decision.changed:
        return Decision(False, f"refusing to promote: {decision.reason}")
    old = _alias_version(client, CHAMPION)
    client.set_registered_model_alias(MODEL, CHAMPION, decision.version)
    client.set_model_version_tag(MODEL, decision.version, "promoted_by", actor)
    if old is not None:
        client.set_registered_model_alias(MODEL, "previous_champion", old.version)
    return Decision(True, f"champion v{old.version if old else '-'} -> "
                          f"v{decision.version} by {actor}; {decision.reason}",
                    decision.version)


def rollback(actor: str, client: MlflowClient | None = None) -> Decision:
    """One command, because the runbook promises one command."""
    client = client or MlflowClient()
    prev = _alias_version(client, "previous_champion")
    if prev is None:
        return Decision(False, "no previous_champion alias to roll back to")
    client.set_registered_model_alias(MODEL, CHAMPION, prev.version)
    client.set_model_version_tag(MODEL, prev.version, "rolled_back_by", actor)
    return Decision(True, f"champion rolled back to v{prev.version} by {actor}",
                    prev.version)
