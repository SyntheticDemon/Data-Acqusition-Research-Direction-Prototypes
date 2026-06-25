# Coding Rules

These rules apply to the entire repository.

## Keep the Code Simple

- Code must be easy for a human reader to inspect and validate.
- Prefer direct, explicit code over abstractions.
- Do not create unnecessary functions, utility functions, tiny three-line
  wrappers, dataclasses, intermediate data contracts, or abstraction layers.
- Do not split code into many modules without a concrete need.
- Keep related code in one file when practical, while organizing that file
  clearly.
- Avoid deeply nested or complex chains of function calls.
- Use clear, descriptive names. Do not use short or cryptic names.
- Introduce a function, class, dataclass, module, or interface only when it
  makes the code materially easier to understand or is required for a real
  variation in the project.

## Interactive Entrypoints

- Do not structure the project around `main()` functions.
- Code should be convenient to run manually from notebooks and Python files
  using `# %%` cells.
- Keep important objects and operations directly accessible so the user can
  import them, inspect them, edit parameters, and execute individual steps.
- Do not hide the research workflow behind a command-line entrypoint.

## Poetry

- This project uses Poetry.
- Run Python commands, scripts, tests, and tools through `poetry run`.
- Do not run dependency installation commands without accounting for Poetry.
- When dependencies change, update `pyproject.toml` as necessary.
- Keep `pyproject.toml` clean and minimal. Do not add unused dependencies,
  tools, metadata, or configuration.

## Visualizations

- Use Plotly Express for visualizations, not Matplotlib.
- Keep plots directly inspectable and convenient to run from notebooks and
  `# %%` cells.

## Project Design

- Design for the real moving parts of this research project.
- The dataset may change.
- The cost model may change.
- The agent or policy implementation may change.
- Keep these parts replaceable without building a large framework around them.
- Use the simplest design that allows these concrete variations.
- Do not generalize for hypothetical requirements beyond the known research
  needs.
