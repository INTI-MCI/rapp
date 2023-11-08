# rapp
Tools for measuring the rotation angle of the plane of polarization (RAPP).


[VNIIOFI]: https://www.vniiofi.ru
[INTI]: https://www.inti.gob.ar/areas/metrologia-y-calidad/fisica/metrologia-fisica

## Introduction

High-accuracy measurements of the rotation angle of the plane of polarization (RAPP)
are needed in many areas of science and engineering [[1]](#1). 
Very few institutes around the world are capable of such measurements.

A new polarimeter will be developed at the Metrology department of ***Instiuto Nacional de Tecnología Industrial*** ([INTI])
based on the research of ***All-Russian Research Institute for Optical and Physical Measurements*** ([VNIIOFI]).

<p align="center">
  <img src="images/diagram.png" />
</p>

In this setup,
> The problem of measuring the RAPP thus reduces
  to measuring the phase difference of two one-dimensional
  harmonic signals [[1]](#1). 

This repository will be used to host code needed to perform the measurements as well as the code
for the post-processing of the output signals and phase difference computations. 

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

## Usage examples:

rapp provides a command-line interface with different commands:

```bash
(.venv) $ rapp
usage: RAPP [-h] {analysis,polarimeter,sim} ...

Tools for measuring the rotation angle of the plane of polarization (RAPP).

positional arguments:
  {analysis,polarimeter,sim}
                        available commands
    polarimeter         Tool for measuring signals with the polarimeter.
    sim                 Tool for making numerical simulations.

options:
  -h, --help            show this help message and exit
```

The polarimeter command:
```bash
(.venv) $ rapp polarimeter -h
usage: RAPP polarimeter [-h] --cycles CYCLES --step STEP --samples SAMPLES [--delay_position] [--analyzer_velocity] [--prefix] [--test] [-v]

options:
  -h, --help            show this help message and exit
  --cycles CYCLES       n° of cycles to run.
  --step STEP           every how many degrees to take a measurement.
  --samples SAMPLES     n° of samples per angle.
  --delay_position      delay (in seconds) after changing analyzer position (default: 1).
  --analyzer_velocity   velocity of the analyzer in deg/s (default: 4).
  --prefix              prefix for the filename in which to write results (default: test).
  --test                run on test mode. No real connections will be established (default: False).
  -v, --verbose         whether to run with DEBUG log level (default: False).

Example: rapp polarimeter --cycles 1 --step 30 --samples 10 --delay_position 0
```

The sim command:

```bash
The sim command:
(.venv) [tlink@tlink rapp]$ rapp sim -h
usage: RAPP sim [-h] [--show] name

positional arguments:
  name        name of the simulation. One of [all, two_signals, error_vs_cycle, error_vs_step, phase_diff].

options:
  -h, --help  show this help message and exit
  --show      whether to show the plot.


```


## References
<a id="1">[1]</a> G. N. Vishnyakov, G. G. Levin, and A. G. Lomakin,
"Measuring the angle of rotation of the plane of polarization by differential polarimetry with a rotating analyzer,"
J. Opt. Technol. 78, 124-128 (2011)
