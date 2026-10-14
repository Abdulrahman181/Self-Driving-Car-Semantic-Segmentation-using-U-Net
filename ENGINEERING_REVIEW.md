# Engineering Review

## Scope

Audited the complete tracked repository for Python, the training notebook, README, dependency declarations, paths, generated artifacts, static correctness, data-split safety, reproducibility, and segmentation metrics. No other repository was modified and no changes were pushed to GitHub.

## Findings and remediation

### Critical / release-blocking before a real run

1. **No dataset or trained model is present.** The historical `Data` file is only a placeholder, and no Cityscapes data or `.keras` model is committed. Training, evaluation, and inference cannot truthfully be run here. The notebook and app now fail early with actionable messages instead of inventing data or metrics.
2. **The original notebook was Kaggle-only and destructive.** It used `/kaggle/input` and `/kaggle/working`, attempted to delete corrupted files from the input tree, and had stale successful-looking outputs mixed with five recorded `AttributeError` failures. It now uses `CITYSCAPES_DATA_ROOT`, validates both splits without deleting source files, writes outputs under `ARTIFACT_DIR`, and has all outputs/execution counts cleared.
3. **The original README documented files that do not exist and claimed unverified results.** References to `src/train.py`, `src/evaluate.py`, `src/predict.py`, `dataset.py`, `results/`, and a validation accuracy of approximately 87% were removed. The README now matches the repository and explicitly states that no benchmark metrics are committed.
4. **The original evaluation derived Dice by transforming mean IoU.** That is not equivalent to a mean of per-class Dice scores. Evaluation now builds a confusion matrix and reports pixel accuracy/loss plus per-class and mean IoU/Dice over classes present in the evaluated set.

### Important engineering fixes

- Added deterministic Python/NumPy/TensorFlow seeds and deterministic `tf.data` options. Platform-level GPU determinism is still not guaranteed.
- Validated non-empty train/validation directories, readable image pairs, and filename overlap to catch a direct train/validation leakage risk. The upstream split is not claimed to be scene-disjoint because scene metadata is unavailable.
- Replaced broad destructive corruption handling with non-destructive validation and explicit errors.
- Added artifact directory handling and write checks for the Keras checkpoint, demo video, and TFLite export.
- Hardened Streamlit model loading with `CITYSCAPES_MODEL`, file existence checks, output-shape/class-count validation, RGB conversion, bounded video processing, temporary-file cleanup, safe class statistics, and one-frame camera snapshots instead of an uninterruptible loop.
- Added bounded runtime requirements and development requirements for Ruff/pytest.
- Expanded `.gitignore` for datasets, model files, generated media, caches, environments, and notebook checkpoints.
- Added `validate_project.py` plus dependency-free static tests.

## Validation performed

All commands below passed in this sandbox:

```text
python3 validate_project.py
python3 -m unittest discover -s tests -v       # 2 tests passed
python3 -m compileall -q stream_segmentation.py validate_project.py tests
ruff check stream_segmentation.py validate_project.py tests
ruff check --select E9,F821,F823 notebook_all_cells.py  # concatenated notebook cells
AST parse of every notebook code cell; no stale outputs/execution counts
```

The final validation is **static only**. TensorFlow training, model loading, video generation, TFLite conversion, and metric reproduction were not executed because the dataset/model artifact are unavailable and heavy ML execution is not assumed.

## Remaining limitations

- The project uses an eight-class, project-specific RGB grouping rather than the complete official Cityscapes taxonomy. The mask encoding must be verified against the supplied image-pairs export before interpreting results.
- Filename-overlap checking catches direct duplicate names only; scene-level leakage requires upstream sequence/scene metadata.
- A real run should archive the exact dependency environment, dataset version, seed/configuration, checkpoint, confusion matrix, and validation outputs before making performance claims.
