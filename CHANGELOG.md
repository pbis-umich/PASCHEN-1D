# Changelog

## 1.1.0 — 2026-09-23

- Added explicit-diffusion stability monitoring alongside drift CFL and made
  adaptive substepping enforce both configured stability targets.
- Added saved diffusion-stability histories, adaptive-limit statistics,
  plotting support, configuration validation, and NumPy/Numba regression
  coverage.
- Connected the energy-dependent Vaughan secondary-electron-yield model to the
  anode boundary flux and added fixed or local-field effective-temperature
  selection.
- Added authenticated BOLSIG+ mean-electron-energy sections and local scalar
  interpolation for the Vaughan temperature closure.
- Corrected the physical diffusion face gradient in every NumPy and Numba
  transport path and added manufactured-solution regression tests.
- Flattened the repository layout and updated CI, packaging, documentation,
  tutorials, examples, and diagnostics for the current source tree.

## 1.0.1 — 2026-09-23

- Corrected the physical diffusion face gradient in the finite-volume
  continuity operator while preserving the Kurganov-Tadmor advective
  reconstruction.
- Added manufactured-solution coverage for public NumPy, low-allocation NumPy,
  Numba serial, and Numba parallel operator paths.

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
