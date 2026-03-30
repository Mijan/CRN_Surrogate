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

- All class members must be **private** (`_` prefix) or **protected** (`_`
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

## Architectural Design Principles

These principles govern how classes and modules are structured. They are not
optional style preferences; they are hard requirements that prevent the kind
of design drift that creates bugs and technical debt.

### Classes must mirror the domain, not the implementation

When designing a class, ask: "what is this thing mathematically / conceptually?"
The fields and methods should match that answer, nothing more.

- If a mathematical object is defined by X and Y, the class should contain X
  and Y. Do not add Z because some algorithm that consumes the object needs Z.
  That algorithm should derive Z from X and Y, or Z should live on the
  algorithm's side.
- Example: a Chemical Reaction Network is defined by stoichiometry + propensity
  functions. A "reactant matrix" is an implementation detail of mass-action
  kinetics and does not belong on the CRN class.

### No free-floating functions for logic that has state or identity

If a function constructs something, transforms something, or dispatches based
on type, it likely belongs as a classmethod, factory method, or method on an
enum. Free functions are acceptable only for pure stateless utilities (math
helpers, tensor manipulation).

- Factory functions that select a subclass based on an enum value should be a
  method on the enum itself, or a classmethod on the base class.
- If a function closes over parameters and returns a callable, make it a
  callable class with `__call__` so the captured parameters are inspectable
  via properties. Raw lambdas are acceptable only for throwaway one-liners in
  tests or examples, never in production code paths.

### Orthogonality: consumers must not know about producers

Modules should depend on interfaces, not on each other. Specifically:

- A simulator (Gillespie, ODE solver) should accept generic inputs (a
  stoichiometry matrix + a callable) not a domain object (CRN). The domain
  object can provide what the simulator needs via a method, but the simulator
  never imports the domain class.
- A neural network encoder should consume a tensor representation, not a
  symbolic domain object. The conversion between symbolic and tensor
  representations happens at an explicit boundary (a dedicated converter
  module), not inside the encoder.
- Notebooks and scripts should use the library's public API (Trainer.train(),
  not a reimplementation of the training loop). If the API is inconvenient,
  fix the API, do not duplicate it.

### No silent fallbacks for missing data

If a loss function requires M >= 2 samples to compute variance, and it
receives M = 1, it must raise a ValueError, not silently return zero. Silent
fallbacks hide bugs. The caller made an error (passed insufficient data) and
must be told.

- Never return a zero tensor as a "safe default" for a loss that cannot be
  computed. This creates a gradient dead zone that is invisible during
  training.
- Never silently squeeze, unsqueeze, or reshape tensors to make shapes
  match. If the input has the wrong number of dimensions, raise a clear error
  stating the expected and actual shapes.

### Parameters belong inside closures, not alongside them

When a function is parameterized (e.g., a propensity function with rate
constants), the parameters should be captured at construction time, not
passed separately at every call site.

- Bad: store `propensity_fn` and `propensity_params` as parallel fields,
  pass params to fn at every evaluation.
- Good: the propensity callable captures its params in a closure. The call
  site just invokes `propensity(state, t)` with no extra arguments.
- The closure should be a callable class (not a raw lambda) so that captured
  parameters are accessible via a `.params` property for serialization and
  debugging.

### Positional indexing into flat tensors is not an API

If a tensor has semantic slots (e.g., params[0] = rate_constant, params[1] =
k_m), the mapping must be defined in exactly one place (a frozen dataclass
with `to_tensor()` / `from_tensor()` methods). No code outside that dataclass
should index into the tensor by position.

### Notebooks must use the library, not reimplement it

If a notebook contains more than 10 lines of logic that duplicates a library
class (e.g., a training loop that mirrors the Trainer), that is a bug. The
notebook should call the library's public API. If the API does not support
what the notebook needs, extend the API.

- Trainer should return a result object that notebooks can plot directly.
- If a notebook needs a custom training variant, subclass the Trainer or
  pass configuration, do not copy-paste the loop.

### Distinguish structural properties from behavioral properties

A domain object often has two kinds of information: what it IS (structure)
and what it DOES (behavior). These should be represented separately, and
derived properties should not be conflated with defining properties.

- If two properties of an object seem related but can diverge in edge cases,
  they are separate concepts and must be stored or computed independently.
- Example: in a reaction network, "which species change when this reaction
  fires" (stoichiometry) and "which species influence how fast this reaction
  fires" (dependency) are distinct. They overlap for mass-action kinetics but
  diverge for catalytic reactions. Never derive one from the other.
- Properties that can be computed from defining properties should be exposed
  as derived `@property` methods, not stored as redundant fields. If
  computation is expensive, use caching.

### Never import private symbols across module boundaries

If a symbol has a leading underscore, it is internal to its module. Other
modules must not import it. This is not a suggestion; it is a hard boundary.

- If module B needs to dispatch on the type of an object created by module A,
  use a protocol (`hasattr` check), a registry keyed on a public type
  (e.g., the params dataclass type), or a method on the object itself. Never
  use `isinstance` checks against private classes from another module.
- If you find yourself importing `_FooBar` from another module, stop. Either
  make it public (remove the underscore and commit to the interface) or
  redesign so the import is unnecessary.

### Prefer computable features over manual annotations for ML inputs

When deciding what information to feed into a neural network encoder, prefer
features that can be derived from the existing data structure over manually
annotated categories.

- Derived features (graph connectivity, stoichiometric coefficients,
  propensity sensitivities) are always consistent with the underlying model,
  cannot be mislabeled, and generalize to novel structures.
- Manual annotations (reaction type labels, biological function categories)
  require domain expertise, are incomplete by nature, and risk the model
  shortcutting structural learning in favor of memorizing labels.
- If the encoder struggles to distinguish important cases from structure
  alone, enrich the computable features (e.g., add edge features, add
  propensity sensitivity estimates) before resorting to annotations.

### Structural constants must have a single source of truth

When a dimension, count, or schema is determined by one module and consumed
by another, define it in exactly one place and have all consumers derive
from that definition. Never duplicate a constant across files, even if both
files agree on the value today.

- If a feature vector has N channels, define an Enum listing the channels in
  the module that constructs the vector. Export `N = len(TheEnum)`. Consumers
  import N rather than hardcoding it.
- Never store structural constants (dimensions determined by data format) in
  config. Config is for hyperparameters (values the user chooses). A dimension
  dictated by the code is not a hyperparameter; it is a derived property.
- If adding a new channel to a feature vector requires changing more than two
  locations (the enum + the builder function), the abstraction is leaking.

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
