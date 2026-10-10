# AGENTS.md — Lab on a Chip revision workstream

These instructions apply to the entire repository when working on branch `lab-on-a-chip-revision`.

## Scientific objective

Rebuild the manuscript around an internally referenced, computationally corrected, closed-loop adaptive microfluidic calorimetry platform. The revised paper should emphasize programmable microfluidics, metrological validation, causal online signal processing, and adaptive selection of subsequent composition setpoints.

Do not treat DNA condensation, DNA photodamage, ctDNA–EB binding, or trypsin kinetics as core evidence in the revised manuscript. Those sections are outside the current revision scope unless the user explicitly restores them.

## Non-negotiable preservation rules

1. Never overwrite or delete raw experimental data.
2. Never rewrite the original submitted manuscript in place. Create revised copies under a dedicated Lab on a Chip revision directory.
3. Do not alter published-result files or model artifacts without preserving the original version and documenting the reason.
4. Do not report new performance numbers until they are regenerated from a locked, leakage-free test set.
5. Do not call retrospective subsampling a closed-loop experiment.
6. Do not use future samples in any online/causal signal-processing path.

## First priority: data integrity audit

Before manuscript rewriting or model tuning, fix the data split and training workflow.

Required properties:

- Split by independent experimental group before creating overlapping windows.
- Use three partitions: train, validation, and locked test.
- The validation set may be used for early stopping, hyperparameter selection, and model-switch thresholds.
- The test set must be used once for final evaluation only.
- No heater event, run, day, chip, or overlapping source interval may appear in more than one partition.
- Produce a machine-readable split manifest containing source identifiers and partition assignments.
- Record random seeds and configuration in version-controlled files.
- Fail loudly if grouping metadata are absent; never silently fall back to random row splitting.

## Target handling

The current MLP applies `log10` to the power target. Before retaining this design, explicitly resolve:

- zero-power samples;
- negative/absorptive heat signals;
- sign restoration;
- whether separate magnitude and sign models are used;
- the relationship between electrical-heater calibration and chemical endothermic measurements.

Do not silently discard non-positive targets.

## Online versus offline processing

Maintain two explicitly named paths:

- `online_causal`: only past and current data; used for actual closed-loop decisions;
- `offline_reference`: may use zero-phase filtering or complete-record processing; used only for retrospective high-precision analysis.

Every closed-loop decision log must record the data available at decision time, fitted parameters, uncertainty, proposed next concentration, command sent to pumps, timestamps, and stop reason.

## Manuscript scope

Recommended main-results sequence:

1. Platform architecture and programmable microfluidic operation;
2. Metrological validation and computational reconstruction;
3. Surfactant micellization thermodynamic mapping;
4. Real closed-loop adaptive composition selection;
5. Generalization and resource-efficiency comparison.

Primary non-biological validation systems should include SDBS plus at least two systems with different curve shapes/signs, preferably one higher-CMC ionic surfactant and TX-100.

## Terminology guardrails

Use conservative language until supported:

- Prefer `non-vacuum, non-hermetic, thermostated benchtop conditions` over `open environment`.
- Prefer `physics-guided calibration` or `computational reconstruction` unless a genuine grey-box physical model is implemented.
- Use `real-time` only after causal processing and latency are measured.
- Use `autonomous selection of subsequent composition setpoints` rather than broad `self-driving laboratory` claims.
- Use `controlled electrical reference input` rather than `ground truth power`.
- Distinguish detection limit from quantification limit.

## Coding expectations

- Use type hints for new Python functions.
- Add clear CLI help and actionable validation errors.
- Separate data I/O, split logic, model training, and evaluation.
- Save configs and manifests alongside outputs.
- Add tests for group exclusivity, deterministic splitting, non-positive targets, and locked-test isolation.
- Preserve backward compatibility only when it does not compromise scientific validity.

## Source of truth

The full revision rationale and task ordering are in:

`docs/lab_on_a_chip_revision/REVISION_MASTER_PLAN.md`

The next executable task is maintained in:

`docs/lab_on_a_chip_revision/CODEX_NEXT_TASK.md`
