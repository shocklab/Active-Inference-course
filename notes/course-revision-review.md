# Course revision, 11 September 2026

Weeks 1 and 2 have been revised following the prose, correctness and reader-depth
review. Jonathan's source notes and private reference material were not edited.

## Changes

- The opening separates feedback regulation from the additional modelling claims
  made by active inference. It retains the stochastic dynamics and variance derivations.
- Inference, learning and action are distinguished. A finite interactive example
  shows that updating beliefs leaves evidence fixed when the model and observation
  are fixed.
- Marginalisation is developed through an observed chain, with both enumeration
  and backward messages. Complexity of enumeration, intermediate factors,
  continuous integration and importance sampling are treated separately.
- The coordinate update for mean-field inference now includes its constrained
  optimisation, first numerical sweep and fixed-point check. A second interactive
  compares factorisation error with sampling variation without claiming to benchmark MCMC.
- The distance model uses an explicitly normalised positive-distance prior.
  Log joint, variational free energy, density transformations, precision units and
  continuous versus discrete updates have distinct definitions.
- A decision example shows how animals with the same posterior can choose
  different actions under different losses. It is explicitly separate from
  expected-free-energy policy selection.
- Headings, captions, controls, notation and notebooks were aligned with these
  changes. Equation numbering now follows reading order through derivation boxes.

## Verification

The build, site structure and computed-number checks pass. The prose and
first-use reports were read as advisory reports; neither is proof of reader clarity.
Separate correctness and reader-depth reviews informed the revisions.

All four notebooks executed locally. Invalid fields on Markdown cells were
removed and the notebook schemas validated. The two Week 2 notebooks were rerun
after their final notation changes. Reference checking resolved all listed DOIs
without metadata mismatches. Independent adaptive quadrature and optimisation
agreed with the distance example's reference calculations within the stated tolerances.

All eleven lesson pages were opened in the browser without KaTeX parse errors.
The four new widgets were exercised, including uninformative sensor settings,
boundary beliefs, repeat sampling and equal-loss decisions. The refined precision
widget agrees with the Python default values at the displayed precision.
The narrow-layout message table was visually inspected.

## Limits

The revised notebooks have not been tested on Colab's free tier. Local execution
is not evidence of Colab compatibility. Weeks 3 to 12 remain outlines.
