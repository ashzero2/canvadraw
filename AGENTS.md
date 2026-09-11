# AGENTS.md — CanvaDraw

Neural network (PyTorch U-Net, trained from scratch) that predicts the next canvas UI frame from current frame + click position. No pre-trained models, no external AI APIs.

Stack: Python 3.12, PyTorch (MPS), Pillow, FastAPI, HTML Canvas.

## Commands

```bash
source .venv/bin/activate
python dataset/generate.py  # regenerate dataset
ruff check .                # lint
ruff format .               # format
python train.py             # train
```

## Phase Workflow

After each phase: stop, report what was done, ask "detailed or concise explanation?", deliver it, then ask "should I commit?".

## Commits

Format: `type(scope): description` — imperative, lowercase, no period, one line, no body, no co-author.
Types: `feat` `fix` `refactor` `chore` `docs`
Scopes: `dataset` `model` `train` `server` `canvas` `config`
No plan or phase references in commit messages — no "phase 1", "phase0", "poc", "plan" etc.

Examples:
```
feat(dataset): add gaussian noise augmentation
fix(train): correct tensor device mismatch on mps
chore(config): add ruff linting rules
```

## Python Rules

- Formatter + linter: Ruff (`pyproject.toml`)
- Run `ruff format .` and `ruff check .` before every commit
- All function signatures must have type hints
- Comments only when the **why** is non-obvious — never describe what the code does
- No unused imports, no unused variables, no placeholder TODOs in committed code
- Device: `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`
