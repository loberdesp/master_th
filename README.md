# master-th

Neural emulation of guitar pedals in PyTorch, targeting deployment on STM32 (master's thesis).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```sh
uv sync                    # create .venv and install all dependencies
uv run pre-commit install  # enable git hooks
```

or simply `make install`. Torch is installed as a CPU-only build (see `[tool.uv.sources]` in `pyproject.toml`).

## Commands

| Command          | What it does                                         |
| ---------------- | ---------------------------------------------------- |
| `make help`      | List all targets                                     |
| `make lint`      | `ruff check` + `ruff format --check`                 |
| `make format`    | Auto-fix lint issues and format                      |
| `make typecheck` | `pyright` (basic mode)                               |
| `make test`      | Fast tests (excludes `slow`, `gpu`, `hardware` marks) |
| `make test-all`  | All tests                                            |
| `make cov`       | Fast tests with coverage report (`htmlcov/`)         |
| `make check`     | lint + typecheck + test                              |
| `make tb`        | TensorBoard on `runs/`                               |
| `make clean`     | Remove caches (never `data/` or `runs/`)             |

Pre-commit runs ruff, basic file hygiene checks, a large-file guard (1 MB),
notebook output stripping and a `uv.lock` consistency check on every commit.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on pushes and PRs to `master`:
`uv sync --frozen`, ruff lint and format check, pyright, and the fast test suite
with a coverage report. Dependabot keeps the workflow's actions up to date monthly.
