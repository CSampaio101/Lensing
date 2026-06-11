# Caustic Lensing and Dark Matter Minihalos

Python simulations of gravitational lensing near fold caustics, including inverse ray shooting, magnification-map generation, and stochastic perturbations from dark matter minihalos.

This repository contains code developed during a Summer 2024 research project supervised by Nikita Blinov. The goal of the project was to study how small-scale dark matter structure can perturb the magnification behaviour of a source near a gravitational-lensing caustic.

A detailed write-up of the project is included in `Lensing Report.pdf`.

<br>

# Project Overview

Gravitational lensing describes the deflection of light by mass. In this project, the main object of interest is the lens equation, which relates the apparent angular position of an image to the true angular position of the source.

Because realistic lens equations are often nonlinear and difficult to solve analytically, this project uses inverse ray shooting. In this method, rays are traced backward from the image plane to the source plane. The magnification is then estimated by comparing the mapping between the two planes.

The project focuses on lensing near a fold caustic, where magnification can become very large. It also adds stochastic perturbations to model the effect of randomly distributed dark matter minihalos.

<br>

# Main Features

- Implements inverse ray shooting for numerical gravitational-lensing calculations
- Generates magnification maps by mapping rays from the image plane to the source plane
- Models lensing near a fold caustic using a Taylor expansion of the Fermat potential
- Adds stochastic deflection perturbations from dark matter minihalos
- Generates Gaussian random fields from a prescribed convergence power spectrum
- Compares numerical magnification curves with analytic and asymptotic expectations
- Produces plots with and without minihalo-induced fluctuations

<br>

# Physics and Numerical Methods

The inverse ray-shooting method divides the image and source planes into pixels. Rays are traced from the image plane back to the source plane using the lens equation. The total magnification is estimated from how the ray mapping changes the apparent flux or area.

Near a fold caustic, the lens equation can be simplified using a Taylor expansion of the Fermat potential. This local approximation makes it possible to study how the magnification of an extended source changes as the source moves toward and across the caustic.

To model dark matter minihalos, the total deflection angle is separated into a smooth component and a stochastic perturbation. The stochastic perturbation is generated from fluctuations in the convergence field, modelled using a Gaussian random field with a chosen power spectrum.

<br>

# Repository Contents

This repository includes code for:

- inverse ray-shooting calculations
- magnification-map generation
- fold-caustic lensing simulations
- stochastic convergence-field generation
- minihalo-induced deflection perturbations
- numerical and analytic magnification-curve comparisons

The accompanying report, `Lensing Report.pdf`, explains the physics background, implementation details, results, unresolved issues, and references.

<br>

# Results

The inverse ray-shooting implementation successfully reproduces expected magnification-map behaviour for test lensing configurations.

The code also produces magnification curves for a Gaussian source moving near a fold caustic. These curves can be generated with and without stochastic perturbations from dark matter minihalos. The minihalo perturbations introduce visible deviations from the smooth-lens magnification curve, illustrating how small-scale dark matter structure can affect caustic-lensing observables.

<br>

# Current Limitations

This repository should be treated as research prototype code rather than a finished software package.

Some discrepancies remain between the numerical magnification curve and the analytic/asymptotic solution, especially near the caustic peak. The report discusses possible sources of these issues, including resolution dependence, source-size effects, and possible normalization issues in the stochastic fluctuation calculation.

Future improvements could include:

- improving numerical convergence near the caustic
- testing resolution dependence more systematically
- implementing additional source brightness profiles
- comparing multiple dark matter density profiles or power spectra
- refactoring the code into a cleaner package structure

<br>

# Dependencies

The code was developed in Python using standard scientific-computing libraries, including:

- NumPy
- SciPy
- Matplotlib

Additional dependencies may be required depending on which script is being run.

<br>

# How to Run

Clone the repository with:

`git clone https://github.com/CSampaio101/Lensing.git`

Then enter the repository:

`cd Lensing`

Install the main dependencies:

`pip install numpy scipy matplotlib`

Run the desired Python script, for example:

`python source.py`

The exact script may depend on which calculation or plot you want to generate.

<br>

# Acknowledgements

This project was supervised by Nikita Blinov in Summer 2024.

The inverse ray-shooting implementation was influenced by Jorge Jimenez-Vicente's tutorial on inverse ray shooting. The Gaussian random field component was adapted from methods inspired by Bruno Sciolla's Gaussian random fields repository.

For the full technical discussion and references, see `Lensing Report.pdf`.

<br>

# Author

Cristiano Sampaio  
MSc Physics, University of Waterloo
