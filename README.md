<h1 align="center" style="border-bottom: none;"> rapp </h1>

<p align="center">
  <a>
    <img alt="Coverage" src="https://codecov.io/gh/tomaslink/rapp/graph/badge.svg?token=DV6XJFKHYS">
  </a>
  <a>
    <img alt="Python versions" src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue">
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

  In this setup,
  > The problem of measuring the RAPP thus reduces
    to measuring the phase difference of two one-dimensional
    harmonic signals [[1]](#1). 

  This repository will be used to host code needed to perform the measurements as well as the code
for the post-processing of the output signals and phase difference computations. 

</div>


> [!IMPORTANT]
> **Disclaimer**: this projects is under an initial development phase. There is no formal release.
  Code can contain serious bugs. 

## How To Contribute

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


## Usage

rapp provides a command-line interface with different commands:
```bash
(.venv) $ rapp
usage: RAPP [-h] [-v] {analysis,OR,phase_diff,plot_raw,polarimeter,sim} ...

Tools for measuring the rotation angle of the plane of polarization (RAPP).

positional arguments:
  {analysis,OR,phase_diff,plot_raw,polarimeter,sim}
                                                  available commands
    analysis                                      Tool for analyzing signals: noise, drift, etc.
    OR                                            Tool for calculating optical rotation from initial phase and final phase measurements.
    phase_diff                                    Tool for calculating phase difference from single polarimeter measurement.
    plot_raw                                      Tool for plotting raw measurements.
    polarimeter                                   Tool for measuring signals with the polarimeter.
    sim                                           Tool for making numerical simulations.

options:
  -h, --help                                      show this help message and exit
  -v, --verbose                                   whether to run with DEBUG log level (default: False).
```

### The polarimeter command

This command allows to acquire harmonic signals from the experimental setup. 

```bash
(.venv) $ rapp polarimeter -h
usage: RAPP polarimeter [-h] [--samples SAMPLES] [--chunk-size] [--reps] [--mc-wait] [--prefix] [--mock-esp] [--mock-pm100] [--disable-pm100] [--work-dir] [-ow] [--mock-adc] [--no-ch0]
                        [--no-ch1] [--cycles CYCLES] [--step STEP] [--delay-position] [--velocity] [--acceleration] [--deceleration] [--hwp-cycles] [--hwp-step] [--hwp-delay-position]

options:
  -h, --help             show this help message and exit
  --samples SAMPLES      n° of samples per angle, default (169) gives 10 cycles of 50 Hz noise.
  --chunk-size           measure data in chunks of this size. If 0, no chunks (default: 500).
  --reps                 Number of repetitions (default: 1).
  --mc-wait              time to wait (in seconds) before re-connecting the MC (default: 15).
  --prefix               prefix for the folder in which to write results (default: test).
  --mock-esp             use ESP mock object. (default: False).
  --mock-pm100           use PM100 mock object. (default: False).
  --disable-pm100        disables the PM100 normalization detector. (default: False).
  --work-dir             folder to use as working directory (default: workdir)
  -ow, --overwrite       whether to overwrite existing files without asking (default: False).

ADC:
  --mock-adc             use ADC mock object. (default: False).
  --no-ch0               excludes channel 0 from measurement (default: False).
  --no-ch1               excludes channel 1 from measurement (default: False).

Analyzer:
  --cycles CYCLES        n° of cycles to run (default: 0).
  --step STEP            step of the rotating analyzer (default: 45). If cycles==0, step is ignored.
  --delay-position       delay (in seconds) after changing analyzer position (default: 0).
  --velocity             velocity of the analyzer in deg/s (default: 4).
  --acceleration         acceleration of the analyzer in deg/s^2 (default: 1).
  --deceleration         deceleration of the analyzer in deg/s^2 (default: 1).

Half Wate Plate:
  --hwp-cycles           n° of cycles of the HW plate (default: 0).
  --hwp-step             motion step of the rotating HW plate (default: 45).
  --hwp-delay-position   delay (in seconds) after changing HW plate position (default: 5).

Example: rapp polarimeter --cycles 1 --step 30 --samples 10 --delay-position 0

```

### The sim command

This command allows to perform simulations to analyze the performance of the phase difference calculation. 

```bash
The sim command:
(.venv) [tlink@tlink rapp]$ rapp sim -h
usage: RAPP sim [-h] [-c] [--name] [--reps] [--angle] [--method] [--cycles] [--step] [--samples] [--dynamic-range] [--max-v] [--noise] [--bits] [--k] [--show]

options:
  -h, --help           show this help message and exit
  -c , --config-file   JSON config file to use (optional).
  --name               name of the simulation to run.

simulation specific parameters:
  --reps               number of repetitions in each simulated iteration.
  --angle              phase difference calculation method.
  --method             phase difference calculation method.
  --cycles             n° of cycles (or maximum n° of cycles) to run.
  --step               motion step of the rotating analyzer.
  --samples            n° of samples per angle.
  --dynamic-range      percentage of dynamic range to simulate (between [0, 1]).
  --max-v              maximum voltage of dynamic range.
  --noise              if True, adds gaussian noise to simulated signals.
  --bits               number of bits for signal quantization.
  --k                  amount of distortion to add to CH0.
  --show               if true, show the plot.

Example: rapp sim --name error_vs_samples --show
```


## References
<a id="1">[1]</a> G. N. Vishnyakov, G. G. Levin, and A. G. Lomakin,
"Measuring the angle of rotation of the plane of polarization by differential polarimetry with a rotating analyzer,"
J. Opt. Technol. 78, 124-128 (2011)
