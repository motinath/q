# QKD System Performance Dataset — Coexistence and Standalone Characterisation

## Overview
This dataset presents QKD (Quantum Key Distribution) performance measurements collected on the TCD/CONNECT quantum-classical coexistence testbed, covering five commercial QKD platforms: a Toshiba MU system, two variants of a Toshiba LE system, IDQ Clavis3, and IDQ ClavisXGR. Measurements span two complementary experimental campaigns run on the same testbed:

- **Coexistence** — QBER and Secure Key Rate (SKR) for the Toshiba MU system  under classical signal injection and varying fibre length, evaluating how quantum and classical traffic coexist on shared optical infrastructure.
- **Characterisation** — full attenuation-sweep QBER/SKR performance for the other four systems running standalone (no coexisting classical signal), plus attenuation-transient (perturbation) response testing on the Toshiba LE research system.

Together these give a full picture of each system's raw performance envelope (Characterisation) and how that performance is affected by realistic classical-quantum coexistence conditions (Coexistence).

## Systems Covered

| System | Measured In | Notes |
|---|---|---|
| Toshiba MU | Coexistence | Used for all coexistence measurements (classical signal injection, fibre-length sweeps). |
| Toshiba LE (production variant) | Characterisation | GHz-clocked efficient decoy-state BB84, phase encoding, InGaAs APD. |
| Toshiba LE (research variant) | Characterisation | Modified for external single-photon detectors; also used for the perturbation/transient-response tests. |
| IDQ Clavis3 | Characterisation | 625 MHz COW protocol. |
| IDQ ClavisXGR | Characterisation | Successor COW-protocol platform to Clavis3. |

## File Listing

### Coexistence files (unprefixed, unchanged from the earlier Zenodo record)
```
baseline_no_signal_no_added_fibre_full_sweep.csv
baseline_no_signal_50km_added_fibre_full_sweep.csv
roadm_3dbm_injected_signal_no_added_fibre_full_sweep.csv
roadm_3dbm_injected_signal_50km_added_fibre_full_sweep.csv
roadm_7dbm_injected_signal_no_added_fibre_full_sweep.csv
roadm_7dbm_injected_signal_50km_added_fibre_full_sweep.csv
roadm_9dbm_injected_signal_no_added_fibre_full_sweep.csv
roadm_9dbm_injected_signal_50km_added_fibre_full_sweep.csv
roadm_12dbm_injected_signal_no_added_fibre_full_sweep.csv
roadm_12dbm_injected_signal_50km_added_fibre_full_sweep.csv
<csv_basename>_stats.json   one per CSV above (e.g. baseline_no_signal_no_added_fibre_full_sweep_stats.json)
csv_summary_stats.json      aggregated stats across all 10 CSVs
```

### Characterisation files (new in this version, prefixed `characterisation_`)
```
characterisation_toshiba_production_full_sweep.csv   full VOA sweep, production system
characterisation_toshiba_production_25samples.csv    same system, 25 samples/point run
characterisation_toshiba_research_3runs.csv          research system, 3 independent runs
characterisation_clavis3_full_sweep.csv              IDQ Clavis3 full VOA sweep
characterisation_clavisxgr_full_sweep.csv            IDQ ClavisXGR full VOA sweep
characterisation_perturbation_baseline.csv           fixed-attenuation baseline, research system
characterisation_perturbation_1s_settling.csv        transient response, 1s settling time
characterisation_perturbation_2s_settling.csv        transient response, 2s settling time
characterisation_perturbation_5s_settling.csv        transient response, 5s settling time
characterisation_perturbation_20s_settling.csv       transient response, 20s settling time
characterisation_perturbation_continuous.csv         transient response, continuous (0s settling)
characterisation_perturbation_continuous_noisy.csv   transient response, continuous, noisy amplitude
characterisation_perturbation_random.csv             transient response, randomised timing/amplitude
characterisation_summary_stats.json                  aggregated summary stats for all files above
```

See `data_dictionary.md` for column definitions (the two groups of files use different schemas, reflecting their different experimental variables) and `methodology.md` for how each dataset was collected and processed.

## Licence
CC-BY 4.0 — see `LICENSE.txt`.

## Citation
If you use this dataset, please cite the Zenodo DOI for this record and reference the IrelandQCI project (Deliverable D4.2).

## Funding
This project has received funding from the European Union's DIGITAL Europe Programme under grant agreement No. 101091520 (IrelandQCI), and from Research Ireland under grants 21/US-C2C/3750 and 13/RC/2077_P2 (CONNECT Centre, Trinity College Dublin).

## Dataset Provenance
This record extends an earlier Zenodo deposit (https://zenodo.org/records/21132088) that published the Coexistence measurements (Toshiba MU system) alone. This version adds the Characterisation campaign in full; standalone attenuation sweeps for the other four systems and the perturbation tests, including the IDQ ClavisXGR system, which had been measured on the testbed but not previously published. The Coexistence data itself is unchanged from the earlier record.
