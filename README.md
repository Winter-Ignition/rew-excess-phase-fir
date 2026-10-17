# REW Excess Phase FIR · v1

English | [简体中文](README.zh-CN.md)

A local workflow and Codex Skill for extracting excess phase from REW measurements, designing near-allpass finite-length FIR correction, and inspecting the exported coefficients. Supports stereo, multiway speakers, unknown crossovers, and paired or independent surround channels.

**After enabling the excess-phase FIR, remeasure and recalibrate the frequency response—even if you already use frequency-response EQ.** Disable the mixed-phase or excess-phase correction being replaced. Preserve essential crossover, protection, and existing system alignment delays.

## Features

- Prefers native impulse-response text or floating-point WAV. Accepts original magnitude/total-phase exports with a complete, dense frequency grid. Precomputed excess-phase input must be declared explicitly.
- Checks impulse peaks and timing conventions; defaults to 1/6-octave complex-vector smoothing to avoid artificial group delay from phase-wrap steps.
- Uses regularized smooth phase inversion from the declared lowest undistorted frequency to 10 kHz. Faster local group-delay correction is limited to the lower of 300 Hz and a conservative estimate based on head movement.
- Fits channels independently. Above 10 kHz, paired channels receive smooth differential correction while preserving common phase.
- Defaults to 192 kHz, 16384 taps, float32 WAV, and 42.67 ms of common reference delay. Audits magnitude error, local out-of-band leakage, and advance energy.
- Produces offline HTML, PNG, CSV, a channel manifest, and independent coefficient acceptance results.

## Use as a Skill

Place `SKILL.md`, `agents/`, `references/`, `scripts/`, and `requirements.txt` in a `rew-excess-phase-fir/` folder inside your Skills directory. Ask a Skill-capable assistant to use `rew-excess-phase-fir` with your REW measurements. Provide channel routing, the lowest undistorted frequency, head movement ranges, and acceptable added delay.

The Skill instructions and detailed references are currently in Simplified Chinese. See [SKILL.md](SKILL.md) and the [measurement/export guide](references/measurement.md).

## Command line

Requires Python 3.10+. Use an isolated environment:

```bash
python -m venv .venv
# Activate your local virtual environment, then run:
python -m pip install -r requirements.txt
python scripts/phase_fir.py inspect left.txt right.txt
python scripts/phase_fir.py design request.json --out design-run
python scripts/phase_fir.py verify design-run
python scripts/test_phase_fir.py
```

See [commands and configuration](references/commands.md) for request JSON, input requirements, and remeasurement comparison, and [algorithm and acceptance](references/algorithm.md) for the method and constraints. Use a new output directory for each design; existing manifests are not overwritten.

## Reading the results

Check low-frequency group-delay peaks and troughs, then actual FIR gain and advance energy. “After correction” curves are predictions until confirmed by acoustic remeasurement. Low-confidence regions are left blank. A single measurement does not establish performance throughout the head movement region. Common reference delay does not align physical speaker/subwoofer arrival times.

Allpass filtering can increase instantaneous audio peaks. Reserve common headroom across channels using the report's conservative bound or actual audio peak tests. Load one complete filter per logical input; avoid stacking old phase correction. Follow the Skill instructions for bass-management placement.

## Tests and privacy

Includes 16 synthetic tests covering minimum phase/pure delay, allpass recovery, unknown three-way crossovers, phase wrapping, input boundaries, routing, coefficient-tamper detection, and separation of global correction from head-movement-limited local detail.

No personal measurements, case reports, machine paths, credentials, or memory data are included. Scripts run locally without automatically uploading data or changing audio device settings. Keep personal inputs and outputs outside the repository or in ignored `measurements/` and `outputs/` directories.

## License

[MIT](LICENSE). References to DRC, REW, and other public sources appear in the algorithm guide. This implementation does not copy or invoke those correction engines and does not claim acoustic comparison against them.
