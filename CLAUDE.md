# CLAUDE.md

## Project Identity

- **Project name**: `TEMPLATE_PROJECT_NAME`
- **Package name**: `TEMPLATE_PACKAGE_NAME`
- **Python version**: `>=3.11`

When initializing a new project from this template, replace ALL occurrences of
`TEMPLATE_PROJECT_NAME` and `TEMPLATE_PACKAGE_NAME` across every file in the
repository (including this file, `pyproject.toml`, `tox.ini`, `setup.cfg`,
GitHub workflow YAMLs, `Makefile`, `README.md`, Dockerfiles, and any
`__init__.py` or config files). Use exact case-sensitive replacement. Confirm
the replacement count with the user before committing.

---

## Code Style

### General Principles

- Write clean, readable code. Prioritize clarity over cleverness.
- Functions should be short (aim for <20 lines in the body). If a function
  does two things, split it into two functions.
- Names must be descriptive and intention-revealing. No abbreviations unless
  they are universally understood in the domain (e.g., `sde`, `crn`, `mlp`).
- No magic numbers or string literals. Extract them into named constants or
  config fields.
- Prefer explicit over implicit. Never rely on side effects for control flow.
- No commented-out code. Remove dead code; git history preserves it.

### Type Annotations

- Type-annotate every function signature, every return type, and every class
  attribute. No exceptions.
- Use `torch.Tensor` (not bare `Tensor`), `list[X]` (not `List[X]`),
  `dict[K, V]`, `tuple[X, ...]` with modern Python syntax.
- Use `Optional[X]` only when `None` is a semantically meaningful value, not
  as a default-avoidance pattern.
- Use `Protocol` or `ABC` for structural/nominal typing of interfaces.
- Generic classes should use `Generic[T]` with meaningful type variables.

### Classes and OOP

- All class members must be **private** (`__` prefix) or **protected** (`_`
  prefix) by default. Only expose through `@property` getters. Provide
  setters only when mutation is explicitly part of the design.
- Use `@dataclass(frozen=True)` for all value objects and configuration.
  Mutable dataclasses need strong justification.
- Use **abstract base classes** (`ABC` + `@abstractmethod`) to define
  interfaces. Concrete implementations inherit from these.
- Use **polymorphism and inheritance** meaningfully. If two classes share
  behavior that varies in a specific dimension, define a base class with the
  shared logic and an abstract method for the varying part.
- Use **composition over inheritance** when the relationship is "has-a"
  rather than "is-a".
- Use `enum.Enum` (or `enum.StrEnum`) for categorical choices, never raw
  strings.
- Each class should have a single responsibility. If a class does two
  unrelated things, split it.
- Implement `__repr__` on all public-facing classes. For dataclasses this is
  automatic.

### Functions

- Functions should do one thing. If the function name contains "and", it
  probably does two things.
- Prefer pure functions (no side effects, deterministic output for given
  input) wherever possible.
- Limit function arguments to 5 or fewer. If more are needed, group them
  into a config or dataclass.
- Use keyword-only arguments (after `*`) for any argument that is not
  self-evident from position.

### Error Handling

- Raise specific exceptions with informative messages. Never use bare
  `except:` or `except Exception:` without re-raising.
- Validate inputs at public API boundaries. Internal functions can trust
  their callers.
- Use custom exception classes for domain-specific errors (inherit from
  appropriate built-in exceptions).

### Package Structure

- Organize code into clearly separated packages and sub-packages. Each
  package should represent a coherent domain concept.
- Every package must have an `__init__.py` that explicitly exports its public
  API via `__all__`.
- Avoid circular imports. If two modules need each other, extract the shared
  abstraction into a third module.
- Keep module files focused. A module with more than ~300 lines likely needs
  splitting.

---

## Documentation

- **Docstrings**: Google style on all public classes, methods, and functions.
  Include `Args:`, `Returns:`, and `Raises:` sections where applicable.
- Private/protected methods should have a brief one-line docstring explaining
  intent, unless the name is fully self-documenting.
- Module-level docstrings at the top of each `.py` file explaining the
  module's purpose in one to three sentences.
- No redundant docstrings that simply restate the function name (e.g.,
  `"""Gets the name."""` on a method called `get_name`).

---

## Testing

- Write unit tests by default for every module. Place them in a parallel
  `tests/` directory mirroring the source structure.
- Use `pytest`. No unittest-style classes unless grouping is genuinely
  needed.
- Test names should describe the behavior being verified, e.g.,
  `test_birth_death_stationary_mean_matches_analytical`.
- Test one behavior per test function. Multiple assertions are fine if they
  verify aspects of the same behavior.
- Use `pytest.fixture` for shared setup. Prefer factory fixtures over
  mutable shared state.
- Use `pytest.mark.parametrize` for testing the same logic across multiple
  inputs.
- Use `pytest.approx` or `torch.testing.assert_close` for numerical
  comparisons. Specify tolerances explicitly.
- Aim for fast tests. If a test needs GPU or takes >5 seconds, mark it with
  `@pytest.mark.slow`.
- Add an integration test that verifies the end-to-end pipeline (data in,
  result out, gradients flow).

---

## PyTorch Conventions

- All `nn.Module` subclasses must call `super().__init__()` first.
- Define sub-modules in `__init__`, compute in `forward`. No module creation
  inside `forward`.
- Name tensor dimensions in comments where shapes are non-obvious, e.g.,
  `# (batch, n_species, d_model)`.
- Use `torch.no_grad()` explicitly for inference and evaluation.
- Prefer `torch.nn.functional` for stateless operations; use `nn.Module`
  wrappers for stateful ones (layers with parameters).

---

## Git Practices

- Commit messages should be imperative and concise: "Add Gillespie SSA
  simulator" not "Added the gillespie simulator".
- One logical change per commit. Do not mix refactoring with feature
  additions.
- Run `tox` (or at minimum `pytest` + linting) before every commit.

---

## Template Replacement Checklist

When creating a new project from this template, perform the following:

1. Replace `TEMPLATE_PROJECT_NAME` with the project's display/repo name in
   all files.
2. Replace `TEMPLATE_PACKAGE_NAME` with the Python package name (lowercase,
   underscores) in all files.
3. Update `pyproject.toml` with correct project metadata (author, description,
   URLs, dependencies).
4. Update `README.md` with project-specific content.
5. Update or remove example/placeholder source files under `src/` or the
   package directory.
6. Update GitHub workflow files with correct package name and any
   project-specific CI steps.
7. Update `tox.ini` with correct package references.
8. Run the full test suite to confirm nothing is broken.
9. Verify no remaining `TEMPLATE_` strings exist anywhere:
   `grep -r "TEMPLATE_" .`
10. Update this section of `CLAUDE.md` to remove the checklist and fill in
    actual project identity above.
