# Methodology

All measurements in this dataset were collected on the same TCD/CONNECT
quantum-classical coexistence testbed, across two complementary experimental
campaigns — see `README.md` for how they relate and for the file naming
convention (Coexistence files are unprefixed; Characterisation files are
prefixed `characterisation_`). Shared equipment: ADVA OPM40 variable optical
attenuators (VOAs), DWDM multiplexers, and circulators.

---

## Coexistence Campaign

### Overview
Measurements were generated using a Toshiba MU QKD system through controlled
attenuation sweeps under different coexistence conditions, evaluating how
classical signal injection and fibre length affect QKD performance.

### Experimental Procedure
- QKD measurements collected over multiple sessions
- Variable Optical Attenuator (VOA) used to perform sweeps
- Each sweep uses fixed noise level (Noise_dBm) and fibre length (Sweep_km)

### Coexistence Scenarios
- Baseline (no external signal)
- Injected classical signals: 3 dBm, 7 dBm, 9 dBm, 12 dBm
- Fibre length: 0 km (back-to-back) or 50 km added fibre

### Data Processing
1. Raw data extraction
2. Cleaning and filtering
3. Grouping by attenuation (VOA)
4. Separation by experimental conditions
5. Generation of summary statistics (JSON)

---

## Characterisation Campaign

### Overview
Standalone (no coexisting classical signal) attenuation-sweep characterisation
of four QKD platforms, plus attenuation-transient response testing, run on
the same testbed as the Coexistence campaign but with the quantum channel
isolated from classical traffic.

### Equipment
- Toshiba LE QKD system (production and research variants)
- ID Quantique Clavis3 QKD system
- ID Quantique ClavisXGR QKD system
- 2x ADVA OPM40 variable optical attenuators (used in parallel <30 dB, series >30 dB)

### Measurement Protocol
1. System allowed to stabilise at each attenuation setting
2. Multiple key blocks collected per operating point (25-100 measurements)
3. SKR and QBER recorded per block
4. Attenuation swept in 0.1 dB increments (production, Clavis3, ClavisXGR) or
   0.1-1.0 dB increments (research system)

### ClavisXGR Data Selection
Due to block fill times on the ClavisXGR, the sweep was captured in two
segments at different sample densities:
1. 100 samples/point, from 6.0-28.9 dB attenuation.
2. 25 samples/point, 29.0-30.6 dB attenuation.
They are combined into a single published file,
`characterisation_clavisxgr_full_sweep.csv`.

### Perturbation Protocol
1. Baseline: 100 measurements at fixed attenuation (9, 12, 15 dB)
2. Perturbations: randomised amplitude (0.1-5.5 dB) applied via VOA to the
   research system
3. Settling times varied: 1s, 2s, 5s, 20s, and continuous (0s)
4. 100 measurements per condition
5. These characterise loss-transient response, not Raman noise transients

### Data Processing
- Raw SNMP readings cleaned of communication timeouts (zero-SKR rows from timeouts removed)
- Attenuation values converted from internal x10 format to dB
- QBER values normalised to fraction (0-1)
- No interpolation or smoothing applied
