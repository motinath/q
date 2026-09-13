# Scientific and Physical Contract for Q-SENTINEL

**System**: ML-Based Diagnostics, Invariant Verification, and Predictive Maintenance for Field-Deployable QKD  
**Protocol Standard**: BB84 with Decoy-State Protocol (ITU-T Y.3800 / ETSI GS QKD 014 / GLLP Framework)  
**Document Classification**: Architectural & Scientific Ground Truth (Layer 0)

---

## 1. Physical System Definition & Operating Baseline

Q-SENTINEL monitors and diagnoses a metropolitan-scale Quantum Key Distribution (QKD) link operating at telecom C-band (1550 nm) over single-mode optical fiber (ITU-T G.652 SMF-28). The physical transceivers and detectors conform to standard commercial specifications:

| Parameter Symbol | Physical Description | Nominal Operating Value | Physical / Mathematical Origin | Unit |
| :--- | :--- | :--- | :--- | :--- |
| $\lambda$ | Optical Carrier Wavelength | $1550.0$ | C-band low-loss optical fiber window | nm |
| $\alpha_0$ | Nominal Fiber Attenuation | $0.20$ | Rayleigh scattering + infrared absorption (SMF-28) | dB/km |
| $L$ | Default Link Distance | $25.0$ | Typical metropolitan inter-node fiber path | km |
| $f_{\text{rep}}$ | Pulse Repetition Rate | $1.0 \times 10^8$ ($100\text{ MHz}$) | Master clock frequency of Alice's pulse laser | Hz |
| $\mu$ | Mean Photon Number (Signal) | $0.50$ | Decoy-state BB84 security optimization | photons/pulse |
| $\nu$ | Mean Photon Number (Decoy) | $0.10$ | Vacuum+Weak decoy-state scheme | photons/pulse |
| $\eta_{\text{bob}}$ | Receiver Optical Efficiency | $0.15$ | Optics transmission $\times$ SPAD quantum efficiency | dimensionless |
| $V_0$ | Nominal Fringe Visibility | $0.985$ | Polarization / phase interferometer contrast | dimensionless |
| $e_{\text{opt}}$ | Intrinsic Optical Error Rate | $0.0075$ | $e_{\text{opt}} = (1 - V_0) / 2$ | fraction |
| $T_0$ | APD Setpoint Temperature | $-40.0$ | Closed-loop Thermo-Electric Cooler (TEC) setpoint | $^\circ\text{C}$ |
| $\text{DCR}_0$ | Nominal Dark Count Rate | $500.0$ | Thermally activated Shockley-Read-Hall carriers | Hz |
| $\tau_{j0}$ | Nominal Timing Jitter | $65.0$ | APD avalanche buildup and time-tagger dispersion | ps |
| $e_{\text{abort}}$ | Critical QBER Security Limit | $0.110$ ($11.0\%$) | Shor-Preskill / GLLP unconditional security threshold | fraction |
| $f_{\text{EC}}(e)$ | Error Correction Efficiency | $1.16$ | Practical Cascade / Multi-edge LDPC efficiency | dimensionless |

---

## 2. Mathematical Physics & Optical Channel Equations (Layer 1)

Every telemetry stream generated or evaluated by Q-SENTINEL adheres strictly to the following canonical equations from quantum optical communications literature.

### 2.1 Optical Channel Transmittance

The channel transmittance $\eta_{\text{channel}}$ through standard fiber of length $L$ with attenuation coefficient $\alpha$ is governed by the Beer-Lambert law:
$$\eta_{\text{channel}} = 10^{-\frac{\alpha \cdot L}{10}}$$
Total channel attenuation $A_{\text{channel}}\text{ [dB]} = \alpha \cdot L$.

### 2.2 Background / Dark Count Probability

The dark count probability per gating window $Y_0$ at clock repetition rate $f_{\text{rep}}$ is:
$$Y_0 = \frac{\text{DCR}}{f_{\text{rep}}}$$

### 2.3 Detector Thermal Generation (Arrhenius Kinetics)

In Avalanche Photodiodes (APDs) operated in Geiger mode (SPADs), thermal carrier generation follows Arrhenius / Shockley-Read-Hall recombination kinetics. For InGaAs/InP detectors, the dark count rate doubles approximately every $\Delta T_{\text{double}} = 10^\circ\text{C}$:
$$\text{DCR}(T) = \text{DCR}_0 \cdot 2^{\frac{T - T_0}{\Delta T_{\text{double}}}}$$

### 2.4 Signal Photon Yield

For an Alice emitting weak coherent pulses with mean photon number $\mu$, the expected signal photon arrival yield $S$ at Bob's detector per pulse is:
$$S = \eta_{\text{channel}} \cdot \eta_{\text{bob}} \cdot \mu$$

### 2.5 Quantum Bit Error Rate (QBER)

The overall QBER ($e$) aggregates background dark count clicks (which yield errors with probability $0.5$), intrinsic optical misalignment $e_{\text{opt}} = \frac{1 - V}{2}$, and eavesdropper interception $\gamma$:
$$\text{QBER} = \frac{\frac{1}{2} Y_0 + e_{\text{opt}} \cdot S + \frac{1}{4} \gamma \cdot S}{Y_0 + S}$$
In the absence of signal ($S \to 0$), $\text{QBER} \to 0.50$ (pure random noise limit). In the ideal limit ($V \to 1.0, Y_0 \to 0$), $\text{QBER} \to 0.00$.

### 2.6 Binary Shannon Entropy

For bit error probability $p \in [0, 1]$:
$$h_2(p) = -p \log_2(p) - (1 - p) \log_2(1 - p)$$
with boundary conditions $h_2(0) = h_2(1) = 0$.

### 2.7 Asymptotic Secret Key Rate (GLLP Decoy-State Framework)

Under the Gottesman-Lo-Lütkenhaus-Preskill (GLLP) and Ma-Qi-Zhao-Lo decoy-state security bounds:
$$R_{\text{SKR}} = \begin{cases} f_{\text{rep}} \cdot \max\left(0, \; Q_1 \left[1 - h_2(e_1)\right] - Q_\mu \cdot f_{\text{EC}}(e) \cdot h_2(e)\right), & \text{if } e < e_{\text{abort}} \\ 0.0, & \text{if } e \ge e_{\text{abort}} \end{cases}$$
where:

- $Q_\mu = Y_0 + 1 - e^{-\eta_{\text{channel}}\eta_{\text{bob}}\mu}$ is the overall gain of the signal state.
- $Q_1 = \mu e^{-\mu} (Y_0 + \eta_{\text{channel}}\eta_{\text{bob}})$ is the gain of the single-photon state.
- $e_1$ is the single-photon phase error rate estimated from decoy states.
- $e_{\text{abort}} = 0.110$ is the Shor-Preskill proof limit for BB84.

### 2.8 Raw Count Rate

The observed total click rate $R_{\text{raw}}$ (counts per second / Hz) is:
$$R_{\text{raw}} = f_{\text{rep}} \cdot (Y_0 + S)$$

---

## 3. Standardized 10-Class Fault & Attack Ontology

Q-SENTINEL establishes an exhaustive, non-overlapping 10-class diagnostic ontology covering physical hardware degradation, channel disturbances, and known quantum optical eavesdropping attacks:

| ID | Class Label | Category | Telecom Severity | Physical Mechanism & Mathematical Signature |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `Normal` | Nominal | `NORMAL` | $V \approx 0.985$, $T \approx -40^\circ\text{C}$, $\text{QBER} \le 0.025$, $\text{SKR} \ge 1.5\text{ kbps}$. |
| 1 | `Optical Misalignment` | Hardware / Fiber | `MEDIUM` | Polarization drift or fiber stress rotation. $V \downarrow$ ($0.985 \to 0.70$), $e_{\text{opt}} \uparrow$. Counts and SNR remain constant; QBER increases linearly with $(1-V)/2$. |
| 2 | `Channel Attenuation Event` | Fiber Plant | `HIGH` | Macro-bending, splice damage, or dirty connector. $\alpha \uparrow$ ($0.20 \to 0.75\text{ dB/km}$). $\eta_{\text{channel}} \downarrow$ exponentially. $R_{\text{raw}}$ collapses, $\text{SNR} \ll 1$, dark counts dominate, driving $\text{QBER} \to 50\%$. |
| 3 | `Detector APD Degradation` | Hardware | `MAJOR` | Trapping centers accumulation in SPAD junction. $\text{DCR} \uparrow$ ($4\times \text{ to } 10\times$). APD temperature $T$ remains nominal ($-40^\circ\text{C}$). Visibility and loss remain nominal. |
| 4 | `Thermal Drift` | Environmental | `MEDIUM` | Thermo-Electric Cooler (TEC) thermal overload or ambient spike. $T \uparrow$ ($-40^\circ\text{C} \to +10^\circ\text{C}$). DCR explodes exponentially via Arrhenius kinetics; strong $\text{Corr}(T, \text{QBER}) > 0.8$. |
| 5 | `Timing Jitter` | Synchronization | `MEDIUM` | Clock phase drift or laser diode driver jitter. $\tau_j \uparrow$ ($65\text{ ps} \to 350\text{ ps}$). Effective coincidence gate window mismatch increases background noise capture. |
| 6 | `Intercept-Resend` | Quantum Attack | `CRITICAL` | Eve intercepts fraction $\gamma$ of pulses and resends in chosen basis (Gisin et al.). Induces $+25\% \cdot \gamma$ error in intercepted pulses. Optical raw counts and loss are unchanged; high QBER occurs despite nominal visibility and dark counts. |
| 7 | `Detector Blinding` | Quantum Attack | `CRITICAL` | Eve shines continuous-wave (CW) bright light into Bob's APDs, forcing them into linear photovoltaic mode (Makarov et al., 2009). Raw click rate explodes ($R_{\text{raw}} > 1.0 \times 10^7\text{ cps}$), measured QBER drops to near zero ($\approx 0.005$), APD temperature remains nominal. |
| 8 | `Photon Number Splitting (PNS)` | Quantum Attack | `CRITICAL` | Eve splits multi-photon pulses on weak coherent pulses ($\mu = 0.50$), suppressing single-photon pulses and storing multi-photons in quantum memory. Decoy-state yield ratio diverges; GLLP security bound collapses, forcing distilled $\text{SKR} = 0.0\text{ bps}$ while QBER remains low ($< 0.05$). |
| 9 | `Time-Shift Attack` | Quantum Attack | `CRITICAL` | Eve introduces basis-dependent sub-nanosecond temporal shifts to exploit detector efficiency mismatch (Zhao et al., 2008). Induces elevated QBER ($0.07 \text{ to } 0.12$) with timing offset ($\tau_j \ge 120\text{ ps}$), with zero extra fiber loss and nominal visibility. |

---

## 4. Physical Invariant Rules (Layer 3 & Layer 5)

ML classifications and telemetry states are evaluated against seven non-negotiable physical invariants:

1. **Invariant 1 (Theoretical Minimum QBER)**:
   $$\text{QBER} \ge \frac{1 - V}{2} - \epsilon_{\text{tol}}$$
   *Rationale*: Total QBER cannot be strictly lower than the intrinsic optical misalignment floor defined by fringe visibility.

2. **Invariant 2 (Dark Count Rate vs Temperature Bound)**:
   $$\text{DCR}_{\text{observed}} \le \text{DCR}_0 \cdot 2^{\frac{T - T_0 + \Delta T_{\text{tol}}}{\Delta T_{\text{double}}}} + \delta_{\text{deg}}$$
   *Rationale*: If temperature is $-40^\circ\text{C}$ and DCR is $4000\text{ Hz}$, the anomaly cannot be attributed to thermal drift (indicates APD trap degradation).

3. **Invariant 3 (Counting Statistics / Transmittance Consistency)**:
   $$\frac{R_{\text{raw}} - \text{DCR}}{f_{\text{rep}} \cdot \mu \cdot \eta_{\text{bob}}} \approx 10^{-\frac{A_{\text{loss}}}{10}}$$
   *Rationale*: Detected photon arrival rate minus dark counts must scale proportionally with channel transmittance.

4. **Invariant 4 (Shor-Preskill Secret Key Rate Bound)**:
   $$\text{If } \text{QBER} \ge 0.110 \implies R_{\text{SKR}} \equiv 0.0\text{ bps}$$
   *Rationale*: No classical error correction and privacy amplification protocol can distill a secure key beyond the Shor-Preskill limit.

5. **Invariant 5 (Detector Blinding Signature)**:
   $$\text{If } R_{\text{raw}} > 1.0 \times 10^7\text{ cps} \text{ and } \text{QBER} < 0.020 \implies \text{Flag CRITICAL Detector Blinding Attack}$$
   *Rationale*: Single-photon counters under normal weak coherent illumination ($\mu=0.5, \text{loss}=5\text{ dB}$) cannot produce $>10\text{ Mcps}$ while maintaining near-zero QBER.

6. **Invariant 6 (PNS Decoy-Yield Contradiction)**:
   $$\text{If } R_{\text{SKR}} == 0.0\text{ bps} \text{ and } \text{QBER} < 0.060 \text{ and } R_{\text{raw}} > 1000\text{ cps} \implies \text{Flag CRITICAL PNS Attack}$$
   *Rationale*: Normal links with low QBER and healthy count rate produce positive SKR unless decoy-state security bounds prove multi-photon interception.

7. **Invariant 7 (Time-Shift Gating Asymmetry)**:
   $$\text{If } \tau_j > 120\text{ ps} \text{ and } \text{QBER} > 0.065 \text{ and } V \ge 0.960 \text{ and } T \le -35^\circ\text{C} \implies \text{Flag CRITICAL Time-Shift Attack}$$
   *Rationale*: Elevated error rate coupled strictly to timing synchronization failure without visibility degradation or thermal heating indicates deliberate gate-phase manipulation.

---

## 5. Formal Optimization Formulation (Layer 8, Rule 9)

Mitigation is **strictly formulated as an argmax optimization** over discrete physical action candidates $\mathcal{A}(c)$ for diagnosed root cause $c \in \{0, \dots, 9\}$, $|\mathcal{A}(c)| \ge 2$.

### Objective Function

For each candidate action $a \in \mathcal{A}(c)$:

1. Project post-action physical state:
   $$\mathbf{x}_{\text{post}}(a) = \mathcal{T}(\mathbf{x}_{\text{pre}}, a)$$
   where $\mathbf{x} = (\alpha, V, T, \text{DCR}, \tau_j)$.
2. Calculate projected post-action $\text{QBER}_{\text{post}}$ and $R_{\text{SKR, post}}$ via Section 2 equations.
3. Compute expected Secret Key Rate Recovery:
   $$\Delta R_{\text{recovery}}(a) = \frac{R_{\text{SKR, post}}(a) - R_{\text{SKR, pre}}}{\max(R_{\text{SKR, baseline}} - R_{\text{SKR, pre}}, \; 1.0)} \times 100\%$$
4. Compute Objective Utility:
   $$J(a) = w_1 \cdot \frac{\Delta R_{\text{recovery}}(a)}{100} - w_2 \cdot \frac{t_{\text{exec}}(a)}{t_{\text{max}}} - w_3 \cdot \mathcal{R}(a)$$
   with weights $w_1 = 0.60, w_2 = 0.20, w_3 = 0.20$, execution time $t_{\text{exec}}$ in seconds, and operational risk score $\mathcal{R}(a) \in [0, 1]$.
5. Optimal Action:
   $$a^* = \arg\max_{a \in \mathcal{A}(c)} J(a)$$
