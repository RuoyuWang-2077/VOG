# Contributing

Keep algorithm changes separate from changes to sampling, random streams, and evaluation protocols. Describe the intended behavior and the affected configuration fields in a pull request.

Run the static and formatting checks before submitting a change:

```bash
python -m pip install -e '.[dev]'
ruff check src
ruff format --check src
```

For model-specific changes, report the base model, reward checkpoints, software versions, device, and observed validation. Do not describe CPU fixtures as full model training results. New configuration fields should have validation, clear defaults, and documentation.
