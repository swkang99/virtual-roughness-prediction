Project: virtual-tactile-property

This repository contains code to train a regressor predicting tactile properties from texture/normal/height images.

Reorganization notes (safe, non-destructive):
- Core library moved/copied to `src/vtp/` for easier packaging and imports.
  - `src/vtp/model.py`, `src/vtp/data.py`, `src/vtp/engine.py`
- Convenience wrapper scripts placed in `scripts/` which set up `sys.path` so imports work without changing* original files.
  - Run via `python scripts/train.py --config config.yaml` etc.

Why this layout?
- `src/` allows packaging or editable installs (`pip install -e .`) without changing import paths across many scripts.
- `scripts/` keeps runnable entrypoints in one place and makes CI easier.

How to run (unchanged workflows still supported):
- Existing root-level scripts still work as before. New wrappers are optional convenience.
- To run training using wrapper:
```cmd
python scripts/train.py --config config.yaml
```

Notes
- This change is non-destructive: files were copied, original files remain.
- If you want, I can now:
  - (A) migrate imports in root scripts to use `src/vtp` and remove duplicates, or
  - (B) make the project installable (add setup/pyproject) so you can `pip install -e .` and import `vtp` directly.

