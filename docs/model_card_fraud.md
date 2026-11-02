# Model card - aegisflow-fraud v1

| Field | Value |
|---|---|
| Alias | champion / challenger |
| MLflow run | logged with train_fraud |
| Algorithm | xgboost / lightgbm + isotonic calibration |
| Training data | IEEE-CIS weeks 1-6 (point-in-time feature matrix), DVC rev train_fraud |
| Evaluation data | weeks 7-8, never touched during training or calibration |
| Headline | recall@1%FPR >= 0.50 (rules baseline 0.38, lift >= 30%) |
| Calibration | ECE <= 0.05 on the test weeks |
| Features | 22 columns, all declared in features/definitions.py |

## Intended use
Ranking payment transactions for approve / review / decline inside AegisFlow, with
the deterministic rule layer able to force a decline regardless of score, and with
SHAP reason codes attached to every decision.

## Out of scope
Any decision about a customer other than this transaction. The score is not a
creditworthiness signal, not a customer risk rating, and not evidence in a SAR -
the SAR narrative is built from the evidence pack, not from the model score.

## Known limitations
- Trained on card-present and card-not-present retail patterns; wire and wallet
  channels are under-represented and their cohort recall is reported separately.
- Laundering typologies are NOT learned by this model; they are detected by the
  graph features and the deterministic typology detectors.
- Cohort screen: Differences across volume channels track base transaction frequency;
  disparate impact screen results are tracked per release.
- Performance decays as rings adapt. The M4 drift job and the label-delay-aware
  rolling metric are what will show it.

## Evaluation
`reports/m3_fraud.html` and `reports/m3_typologies.html`, both logged on the run.
