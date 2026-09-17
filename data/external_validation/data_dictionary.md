# Data Dictionary

This dataset covers two experimental campaigns from the same testbed (see
`README.md`), each with its own CSV schema reflecting its own experimental
variables. Zenodo does not support folders, so files are distinguished by
name: Coexistence files are unprefixed, Characterisation files are prefixed
`characterisation_`.

---

## Coexistence CSV Variables (unprefixed `*.csv` files)

All files in this group were collected on the Toshiba MU system; there is no
`System` column since it is constant across the group.

### Noise_dBm
- External signal power injected via the AUX input
- Values: "none" (no signal), or numeric dBm (3, 7, 9, 12)

### Sweep_km
- Fibre length
- Values: 0 (no added fibre, back-to-back), 50 (50 km added fibre)
- Unit: km

### VOA_dB
- Optical attenuation applied
- Main sweep variable
- Unit: dB

### QBER
- Quantum Bit Error Rate
- Unit: fraction (0 to 1)

### SecureKeyRate_bps
- Secure Key Rate
- Unit: bits per second

### Data Behaviour
- Increasing attenuation: QBER increases, SKR decreases
- Increasing injected noise: QBER increases, SKR decreases faster
- At high attenuation: SKR → 0

### Coexistence JSON Summary Files (unprefixed `*.json` files)
- One file per CSV (`<csv_name>_stats.json`) plus one aggregated
  `csv_summary_stats.json`
- Fields: row_count, and per-VOA-block QBER/SKR stats (mean, min, max, median,
  std_dev), zero-SKR counts/fractions, and a `fully_collapsed` flag

---

## Characterisation CSV Variables (`characterisation_*.csv` files)

### System
- QKD platform identifier
- Values: `Toshiba_LE_Production`, `Toshiba_LE_Research`, `IDQ_Clavis3`, `IDQ_ClavisXGR`

### Fibre_km
- Fibre spool length (where applicable)
- Values: 0 (back-to-back)
- Unit: km

### VOA_dB
- Variable optical attenuator setting
- Main sweep variable
- Unit: dB

### QBER
- Quantum Bit Error Rate
- Unit: fraction (0 to 1)

### SecureKeyRate_bps
- Secure Key Rate
- Unit: bits per second

### Run (`characterisation_toshiba_research_3runs.csv` only)
- Independent measurement run index (1, 2, or 3)

### Condition (`characterisation_perturbation_*.csv` only)
- Perturbation condition label
- Values: baseline, 1s_settling, 2s_settling, 5s_settling, 20s_settling,
  continuous, continuous_noisy, random

### Characterisation Summary File (`characterisation_summary_stats.json`)
- One entry per dataset (production, production_25, research, clavis3,
  clavisxgr, and one per perturbation condition)
- Fields: total_records, positive_skr_records, voa_range_dB, mean/min/max/std
  SKR_bps, mean/min/max QBER
