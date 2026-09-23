# Changelog

## 1.0.1 — 2026-09-23

- Corrected the physical diffusion face gradient in the finite-volume
  continuity operator. Version 1.0.0 subtracted two slope-limited
  Kurganov-Tadmor reconstructions at the same face; version 1.0.1 differences
  adjacent cell-centered densities while leaving the advective reconstruction
  unchanged.
- Added manufactured-solution regression coverage for the public NumPy,
  low-allocation NumPy, Numba serial, and Numba parallel operator paths.
- Results and checkpoints produced by version 1.0.0 remain part of the
  historical numerical lineage and should not be relabeled as corrected
  version 1.0.1 output.

## 1.0.0 — 2026-07-21

- Added strict manifest-backed BOLSIG+ electron transport for 42 neutral gases.
- Added normalized LXCat positive-ion mobility and longitudinal-diffusion tables, compatible-pair validation, portable data paths, and table provenance.
- Added the complete ion-transport configuration model to every standalone
  configuration module.
- Made every supplied case configuration a complete standalone dataclass
  definition with case-tuned defaults and schema-parity regression coverage.
- Documented every public selector's accepted options beside its field.
- Removed the unsupported electron-energy PDE path, ambiguous circuit aliases, and circuit auto-detection fallbacks.
- Added fail-fast validation for public selectors, numerical limits, physical state, boundary/emission combinations, circuits, and swarm-table identity.
- Added E/N coverage diagnostics, software/dependency provenance, source checksums, and output-reader support.
- Added circuit, corrupt-table, solver, backend-parity, output round-trip, and miniature golden-case regression tests.
- Added reproducible dependency locks, source-distribution metadata, CI, release-archive tooling, and updated documentation.
