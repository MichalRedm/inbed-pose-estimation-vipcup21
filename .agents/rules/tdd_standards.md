# Test-Driven Development (TDD) Standards

This repository strictly follows Test-Driven Development (TDD) principles to ensure the stability and reliability of the pose estimation pipeline.

## 1. Core Principles
- **Red-Green-Refactor**: Always write a failing test before implementing a new feature or fixing a bug.
- **Coverage**: Maintain high test coverage for core logic, especially in `src/training/`, `src/models/`, and `src/data/`.
- **Atomic Tests**: Each test should focus on a single unit of functionality.

## 2. Testing Requirements
- **New Features**: Every new feature MUST be accompanied by a unit test in the `tests/` directory.
- **Bug Fixes**: A regression test MUST be added to prevent the bug from reappearing.
- **Unified Training**: Any changes to the trainer factory or new training modes MUST include tests verifying correct initialization and loss calculation.

## 3. Automation
- **Pre-commit**: Run `pytest` before every commit.
- **CI/CD**: The CI pipeline will reject any PR that does not pass the full test suite.

## 4. Documentation
- Use docstrings to explain the purpose of each test.
- Keep tests readable and well-structured.
