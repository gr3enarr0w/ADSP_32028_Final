<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/395806356 -->
<!-- Last synced: 2026-05-29 -->

# ML Operations

## Triggering a Manual Retrain

```bash
oc create job --from=cronjob/ai-helpdesk-retrain manual-retrain-$(date +%s) \
  -n jira-messaging--runtime-ext
oc logs job/manual-retrain-<timestamp> -n jira-messaging--runtime-ext -f
```

## Interpreting validation_report.json

The file at `models/validation_report.json` is written by `scripts/validate_models.py`. Key fields:

- `holdout.ensemble.macro_f1` — primary metric; must exceed 0.70 for promotion
- `cv.ensemble.mean` / `cv.ensemble.std` — cross-validation stability; std > 0.05 warrants investigation
- `calibration.ece` — Expected Calibration Error; < 0.20 is acceptable
- `learning_curves` — F1 at 10–80% training data; should increase monotonically

## What to Do When the Promotion Gate Rejects a Model

A rejected retrain is **expected and correct** — it means the new model didn't improve. Check the structured JSON log for `new_macro_f1` vs `prev_macro_f1`. If the gap is large (> 0.05), investigate label noise: the most recent `classify_unclassified()` run may have produced poor labels.

To manually inspect: `python -m scripts.validate_models --quick`

## Production Baseline

Current: Macro-F1 **0.7075**, Accuracy **89.8%**, trained on 6,705 tickets (80/10/10 split).
Stored in `models/production_metrics.json` on the `ai-helpdesk-agent-data` PVC.
