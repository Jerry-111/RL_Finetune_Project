# WOSAC Evaluation Report (2026-02-13)

## Objective
Validate that PufferDrive WOSAC evaluation runs end-to-end on our environment and produces stable realism metrics.

## Environment
- Repo: `PufferDrive` (branch: `2.0`)
- Python env: `.venv`
- Device: `cuda`
- Policy checkpoint: `resources/drive/pufferdrive_weights.pt`

## Dataset
- Source dataset: `daphne-cornelisse/pufferdrive_womd_val` (Hugging Face)
- Downloaded archive: `pufferdrive_val.zip`
- Extracted JSON path: `/jerry_slow_vol/pufferdrive_womd_val_raw/data/processed/validation`
- JSON validation: `10000` valid, `0` invalid (`post_processing.py`)
- Converted binaries path: `/jerry_slow_vol/pufferdrive_womd_val_bins/validation`

## Why Not `avl-west` Directly
- The available `avl-west` data for Waymo was in TFRecord form.
- PufferDrive WOSAC eval expects PufferDrive map binaries (`map_*.bin`) in `--eval.map-dir`.
- In this repo, the provided conversion script is JSON -> `map_*.bin` (not TFRecord -> `map_*.bin`).
- Therefore, we used a prepared JSON-based dataset, validated it, and converted it to binaries with their preprocessing code before evaluation.

## Evaluation Command
```bash
puffer eval puffer_drive \
  --eval.wosac-realism-eval True \
  --eval.map-dir "$BIN_DIR" \
  --eval.wosac-target-scenarios 229 \
  --eval.wosac-scenario-pool-size 229 \
  --eval.wosac-num-rollouts 32 \
  --eval.wosac-batch-size 32 \
  --eval.wosac-max-batches 100 \
  --train.device cuda \
  --load-model-path /root/PufferDrive/resources/drive/pufferdrive_weights.pt
```

## Runtime Summary
- Completed with no crash.
- Progress reached target coverage (`n=229`) in `76` batches.
- Duplicate sampled scenarios dropped: `731`.
- Final unique scenarios in aggregated result: `211`.

## Final Metrics
- `realism_meta_score`: **0.6766350710900473**
- `realism_meta_score_std`: `0.11832249917822324`
- `kinematic_metrics`: `0.2760169769611165`
- `interactive_metrics`: `0.6482248371829668`
- `map_based_metrics`: `0.7890867266461042`
- `ade`: `9.817255924170617`
- `min_ade`: `9.201450236966824`
- `num_collisions_sim`: `0.14117061611374407`
- `num_collisions_ref`: `0.05260189573459716`
- `num_offroad_sim`: `0.15771563981042658`
- `num_offroad_ref`: `0.16228436018957346`
- `total_num_agents`: `611`
- `total_unique_scenarios`: `211`

## Interpretation
- The WOSAC pipeline is functional and produces coherent realism metrics.
- Overall realism score is close to the baseline scale reported in repo docs.
- Exact values are not expected to match table values in docs because this run used the converted `pufferdrive_womd_val` setup rather than the curated clean WOSAC subset.

## Notes
- This report demonstrates operational correctness (data prep, conversion, inference, metric aggregation).
- For strict apples-to-apples comparison with the docs table, run on the same curated clean scenario subset and identical evaluation protocol.
