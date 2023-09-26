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

Then, for example, run the code in the cli:
```bash
python cli.py
```


## References
<a id="1">[1]</a> G. N. Vishnyakov, G. G. Levin, and A. G. Lomakin,
"Measuring the angle of rotation of the plane of polarization by differential polarimetry with a rotating analyzer,"
J. Opt. Technol. 78, 124-128 (2011)
