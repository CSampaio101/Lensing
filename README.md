````markdown
# Caustic Lensing and Dark Matter Minihalos

This repository contains Python code developed during a Summer 2024 research project supervised by Nikita Blinov. The project studies how small-scale dark matter structure, modelled as randomly distributed dark matter minihalos, can perturb magnification behaviour near gravitational-lensing caustics.

The code implements numerical tools for gravitational lensing near fold caustics, including inverse ray shooting, magnification-map generation, and stochastic convergence fluctuations generated from a prescribed power spectrum.

## Project Overview

Gravitational lensing is described by the lens equation

\[
\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta}),
\]

where \(\boldsymbol{\beta}\) is the angular position of the source, \(\boldsymbol{\theta}\) is the observed image position, and \(\boldsymbol{\alpha}\) is the deflection angle. Because the lens equation is generally nonlinear, numerical methods are often required for realistic lensing models.

This project focuses on caustic lensing, where magnification can become very large near special curves called caustics. The goal was to model how small-scale dark matter fluctuations can alter the expected magnification curve near a fold caustic.

## Main Features

- Implements inverse ray shooting for solving lensing configurations numerically
- Generates magnification maps by mapping rays from the image plane to the source plane
- Models lensing near a fold caustic using a Taylor expansion of the Fermat potential
- Adds stochastic deflection perturbations from dark matter minihalos
- Generates Gaussian random fields from a chosen convergence power spectrum
- Compares numerical magnification curves with analytic/asymptotic expectations
- Produces plots with and without minihalo-induced fluctuations

## Physics Background

The inverse ray-shooting method maps rays from the image plane back to the source plane. The magnification is then estimated by comparing the flux or area mapping between the two planes.

Near a fold caustic, the lens equation can be simplified by expanding the Fermat potential around a point on the caustic. The resulting local lens mapping is used to study how the magnification of an extended Gaussian source changes as the source moves across the caustic.

To include the effect of dark matter minihalos, the total deflection angle is written schematically as

\[
\boldsymbol{\alpha} = \boldsymbol{\alpha}_S + \delta \boldsymbol{\alpha},
\]

where \(\boldsymbol{\alpha}_S\) is the smooth lensing contribution and \(\delta \boldsymbol{\alpha}\) is a stochastic perturbation sourced by fluctuations in the convergence field. These convergence fluctuations are generated using a Gaussian random field with a prescribed power spectrum.

## Repository Contents

The repository includes code for:

- inverse ray shooting
- magnification-map generation
- fold-caustic lensing calculations
- stochastic fluctuation generation
- plotting numerical and analytic magnification curves

A detailed write-up of the project is included in `Lensing Report.pdf`.

## Results

The inverse ray-shooting implementation successfully reproduces expected magnification-map behaviour for test lensing configurations. The code also produces magnification curves for a Gaussian source moving near a fold caustic, both with and without stochastic perturbations from dark matter minihalos.

The minihalo fluctuations introduce visible deviations from the smooth-lens magnification curve, illustrating how small-scale dark matter structure can affect caustic-lensing observables.

## Current Limitations

This repository should be treated as research/prototype code. Some discrepancies remain between the numerical magnification curve and the analytic/asymptotic solution, especially near the caustic peak. The report discusses possible sources of these issues, including resolution dependence, source-size effects, and normalization of the stochastic fluctuation calculation.

Future improvements could include:

- improving numerical convergence near the caustic
- testing resolution dependence more systematically
- implementing additional source brightness profiles
- comparing multiple dark matter density profiles or power spectra
- refactoring the code into a cleaner package structure

## Dependencies

The code was developed in Python and uses standard scientific-computing libraries, including:

- NumPy
- SciPy
- Matplotlib

Additional dependencies may be required depending on which scripts are run.

## How to Run

Clone the repository:

```bash
git clone https://github.com/CSampaio101/Lensing.git
cd Lensing
````

Install the required Python packages:

```bash
pip install numpy scipy matplotlib
```

Then run the desired script, for example:

```bash
python source.py
```

Update the command above depending on the script you want to run.

## Acknowledgements and References

This project was supervised by Nikita Blinov in Summer 2024.

The inverse ray-shooting implementation was influenced by Jorge Jimenez-Vicente's tutorial on inverse ray shooting. The Gaussian random field component was adapted from methods inspired by Bruno Sciolla's Gaussian random fields repository.

For a full explanation of the physics, implementation details, results, and references, see `Lensing Report.pdf`.

## Author

Cristiano Sampaio
MSc Physics, University of Waterloo

```
```
