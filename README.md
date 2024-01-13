<h1 align="center" style="border-bottom: none;"> rapp </h1>

<p align="center">
  <a>
    <img alt="Coverage" src="https://codecov.io/gh/tomaslink/rapp/graph/badge.svg?token=DV6XJFKHYS">
  </a>
</p>

Tools for measuring the rotation angle of the plane of polarization (RAPP).


[VNIIOFI]: https://www.vniiofi.ru
[INTI]: https://www.inti.gob.ar/areas/metrologia-y-calidad/fisica/metrologia-fisica

## Introduction

<div align="justify">

  High-accuracy measurements of the rotation angle of the plane of polarization (RAPP)
  are needed in many areas of science and engineering [[1]](#1). 
  Very few institutes around the world are capable of such measurements.

  A new high-accuracy polarimeter is being developed at the Physics Metrology department
  of ***Instituto Nacional de Tecnología Industrial*** ([INTI])
    based on the research
  of ***All-Russian Research Institute for Optical and Physical Measurements*** ([VNIIOFI]).

  <p align="center">
    <img src="images/diagram.png" />
  </p>

  In this setup,
  > The problem of measuring the RAPP thus reduces
    to measuring the phase difference of two one-dimensional
    harmonic signals [[1]](#1). 

  This repository will be used to host code needed to perform the measurements as well as the code
for the post-processing of the output signals and phase difference computations. 

</div>

## How To Contribute

All binary files like images must be stored using GIT LFS (e.g. `git lfs track "*.png"`).

0. Install Git LFS from [here](https://git-lfs.com).

1. Clone the repository: 

```bash
  git clone https://github.com/INTI-metrologia/rapp
```

2. Create virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run tests:

```bash
pytest
```


## Usage examples:

rapp provides a command-line interface with different commands:

```bash
(.venv) $ rapp
usage: RAPP [-h] {polarimeter,phase_diff,or,analysis,sim} ...

Tools for measuring the rotation angle of the plane of polarization (RAPP).

positional arguments:
  {polarimeter,phase_diff,or,analysis,sim}
                        available commands
    polarimeter         Tool for measuring signals with the polarimeter.
    phase_diff          Tool for calculating phase difference from single polarimeter measurement.
    or                  Tool for calculating optical rotation from initial phase and final phase measurements.
    analysis            Tool for analyzing signals: noise, drift, etc.
    sim                 Tool for making numerical simulations.

options:
  -h, --help            show this help message and exit
```

The polarimeter command:
```bash
(.venv) $ rapp polarimeter -h
usage: RAPP polarimeter [-h] --samples SAMPLES [--cycles CYCLES] [--step STEP] [--chunk-size] [--delay-position] [--velocity] [--init-position] [--hwp-cycles] [--hwp-step] [--hwp-delay] [--reps] [--no-ch0] [--no-ch1] [--prefix]
                        [--mock-esp] [--mock-adc] [--plot] [-v] [-w]

options:
  -h, --help         show this help message and exit
  --samples SAMPLES  n° of samples per angle.
  --cycles CYCLES    n° of cycles to run (default: 0).
  --step STEP        motion step of the rotating analyzer (default: 45).
  --chunk-size       measure data in chunks of this size. If 0, no chunks (default: 500).
  --delay-position   delay (in seconds) after changing analyzer position (default: 1).
  --velocity         velocity of the analyzer in deg/s (default: 4).
  --init-position    initial position of the analyzer in deg (default: None).
  --hwp-cycles       n° of cycles of the HW plate (default: 0).
  --hwp-step         motion step of the rotating HW plate (default: 45).
  --hwp-delay        delay (in seconds) after changing HW plate position (default: 5).
  --reps             Number of repetitions (default: 1).
  --no-ch0           excludes channel 0 from measurement (default: False).
  --no-ch1           excludes channel 1 from measurement (default: False).
  --prefix           prefix for the filename in which to write results (default: None).
  --mock-esp         use ESP mock object. (default: False).
  --mock-adc         use ADC mock object. (default: False).
  --plot             plot the results when the measurement is finished (default: False).
  -v, --verbose      whether to run with DEBUG log level (default: False).
  -w, --overwrite    whether to overwrite existing files without asking (default: False).

Example: rapp polarimeter --cycles 1 --step 30 --samples 10 --delay-position 0
```

The sim command:
```bash
The sim command:
(.venv) [tlink@tlink rapp]$ rapp sim -h
usage: RAPP sim [-h] [--samples SAMPLES] [--step STEP] [--reps REPS] [--show] [-v] name

positional arguments:
  name               name of the simulation. One of ['all', 'error_vs_method', 'error_vs_step', 'error_vs_range', 'error_vs_samples', 'error_vs_res', 'signals_out_of_phase', 'sim_steps', 'noise_vs_range', 'phase_diff'].

options:
  -h, --help         show this help message and exit
  --samples SAMPLES  n° of samples per angle.
  --step STEP        n° of samples per angle.
  --reps REPS        number of repetitions in each simulated iteration (default: 1).
  --show             whether to show the plot.
  -v, --verbose      whether to run with DEBUG log level (default: False).

Example: rapp sim error_vs_samples --show
```


## References
<a id="1">[1]</a> G. N. Vishnyakov, G. G. Levin, and A. G. Lomakin,
"Measuring the angle of rotation of the plane of polarization by differential polarimetry with a rotating analyzer,"
J. Opt. Technol. 78, 124-128 (2011)
