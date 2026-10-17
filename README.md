# REW Excess Phase FIR · v2

English | [简体中文](README.zh-CN.md)

Local excess-phase analysis and near-allpass FIR design for REW measurements. Supports stereo, unknown and multiway crossovers, paired surround channels, independent channels, and aligned subwoofer systems. Author: **Winter-Ignition**.

**After enabling the FIR, remeasure and recalibrate frequency-response EQ, including any EQ already in use.** Preserve essential crossovers, protection and physical alignment delays. Disable the mixed-phase or excess-phase correction being replaced when making a fresh baseline measurement.

## Features

- Extracts excess phase from native impulse responses or dense magnitude/total-phase exports. Defaults to 1/6-octave complex-vector smoothing, with wrap and window sensitivity checks.
- Separates the speaker's lowest undistorted frequency from the measurement's noise-dominated lower boundary. Fits an equivalent low-end magnitude extension without using observations below that boundary; records alternative-model sensitivity.
- Solves broad phase correction across the usable lower bound to 10 kHz. Faster local group-delay detail is restricted to the lower of 300 Hz and a head-movement estimate; this local limit does not restrict broad correction.
- Fits channels independently. Smoothly joins the common correction above 10 kHz and reduces paired-channel differences near 10 kHz, without forcing all higher-frequency excess phase to zero. Exit frequencies follow available trusted bandwidth.
- Can revise measurements made with a previous FIR. Requires the actual loaded coefficients, hash, channel mapping and common delay; exports one complete replacement rather than another layer.
- Defaults to 192 kHz, 16384 taps, float32 WAV and 42.67 ms common reference delay. Reads exported coefficients back for gain, advance-energy, transition, composition and symmetry checks, including direct DTFT cross-checks.
- Generates offline HTML, PNG, CSV, a channel manifest and coefficient acceptance results. Acoustic remeasurement uses the original low-end reference and smoothing convention.

## Measurement example

Real listening-position measurements from one stereo system, with 1/6-octave vector smoothing and analysis-only timing reference removal. The gray dashed curves come from the retained initial measurement analysis; the blue curves are the final acoustic remeasurement, not a prediction. The lower panel enlarges the remaining phase variation.

![Initial and final measured excess phase for both channels, with an enlarged view of the final residuals from 200 Hz to 10 kHz.](docs/images/measured-excess-phase.png)

The low-frequency reference was refined during the work, so the plot below shows the final remeasurement on its own. It uses a fitted low-end reference; dotted segments indicate lower vector coherence. Room-related structure and measurement noise remain visible.

![Final measured excess group delay from 38 to 300 Hz for the left and right channels, including the remaining peaks and troughs.](docs/images/measured-low-frequency.png)

The actual exported float32 coefficients stay within approximately ±0.011 dB in this example. The impulse plot shows both sides of the common reference time, including advance energy.

![Actual FIR magnitude within the design limits and left/right impulse responses around the 42.67 ms common reference delay.](docs/images/fir-coefficient-checks.png)

## Requirements and setup

Python 3.10+, NumPy ≥1.24, SciPy ≥1.10, SoundFile ≥0.12 and Matplotlib ≥3.7. Use a convolution host that supports the exported format and channel layout.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python scripts/phase_fir.py inspect left.txt right.txt
python scripts/phase_fir.py design request.json --out design-run
python scripts/phase_fir.py verify design-run
python scripts/phase_fir.py compare remeasure.json --reference design-run --out remeasure-run
python -m unittest discover -s scripts -p "test_*.py"
```

Use a new output directory for each design. See [configuration and commands](references/commands.md), [measurement/export](references/measurement.md) and [algorithm/acceptance](references/algorithm.md). Detailed Skill instructions, references and reports are currently in Simplified Chinese.

To install as a Skill, place `SKILL.md`, `agents/`, `references/`, `scripts/` and `requirements.txt` in a `rew-excess-phase-fir/` folder in your Skills directory. Supply channel routing, crossover information if known, head movement half-ranges, the lowest undistorted frequency, the trustworthy measurement cutoff, and the allowed added delay.

## Scope and limits

Finite FIR correction is approximately allpass. Available delay and advance-energy constraints limit the correction; deep notches and inconsistent phase receive conservative treatment. The low-end rolloff is an equivalent reference model, not identification of a physical enclosure. An unknown noise boundary leaves low-frequency absolute excess phase unverified. Full-band export alone does not make every point trustworthy; noise-dominated ultrasonic FRD is analyzed using a declared audible-band reference.

Plots marked “after” are coefficient predictions until acoustic remeasurement. Compare phase and group delay by band: a lower phase RMS does not establish that every group-delay peak improved. A single listening-position measurement cannot establish stability across a head movement region. This tool does not align physical speaker arrivals or configure bass management. Independent IR shifts do not recover real channel delays.

Load one complete filter per logical input. A replacement includes the previous FIR once and keeps one common reference delay; remove the old filter. Bass-managed branches of the same logical input must share correction. Reserve common digital headroom because phase changes can raise audio peaks.

## Versions and data

[v1 remains available](https://github.com/Winter-Ignition/rew-excess-phase-fir/releases/tag/v1). v2 changes the high-frequency handoff and adds noise-aware low-end reference modeling and complete-replacement design. Use v2 manifests and analysis files together; v1 coefficient files can be supplied as explicitly mapped loaded filters, but v1 reports are not v2 reference packages.

Scripts run locally without uploading measurements or changing audio-device settings. No raw personal measurements are included. The README images are owner-authorized measurement and filter plots with identifying details removed; source files and analysis records remain private. Keep measurements, reports and generated filters outside the checkout, or in ignored local directories.

## License

[MIT](LICENSE), Winter-Ignition. Public sources and method boundaries are listed in the algorithm guide.
