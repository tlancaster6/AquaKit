# Contributing

## Development Setup

```bash
# Clone the repository
git clone https://github.com/tlancaster6/aquacore.git
cd aquacore

# Install Hatch (build/environment manager)
pip install hatch

# Create the default development environment
hatch env create

# Install pre-commit hooks
hatch run pre-commit install
hatch run pre-commit install --hook-type pre-push
```

## Running Tests

```bash
hatch run test                                    # fast tests (skip @pytest.mark.slow)
hatch run test-all                                # all tests
hatch run pytest tests/unit/test_smoke.py -v     # single file
```

## Code Quality

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
hatch run lint                # lint
hatch run lint -- --fix       # lint + auto-fix
hatch run format              # format
```

## Type Checking

This project uses [basedpyright](https://docs.basedpyright.com/) for type checking:

```bash
hatch run typecheck           # run type checker
hatch run check               # lint + typecheck
```

Pre-commit hooks run Ruff automatically on each commit. Pre-push hooks run
the full linter and formatter check.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` -- new feature (triggers minor version bump)
- `fix:` -- bug fix (triggers patch version bump)
- `docs:` -- documentation only
- `test:` -- adding or updating tests
- `refactor:` -- code restructuring without behavior change
- `ci:` -- CI/CD changes
- `chore:` -- maintenance tasks

## Building Documentation

```bash
hatch run docs:build
```
