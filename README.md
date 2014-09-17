Hamiltonian Annealed Importance Sampling (HAIS)
================================

HAIS is a method for combining Hamiltonian Monte Carlo (HMC) and Annealed Importance Sampling (AIS), 
so that a single Hamiltonian trajectory can stretch over many AIS intermediate distributions. This 
greatly improves the efficiency of AIS in continuous state spaces.  It is described in detail in the paper:
> J Sohl-Dickstein, BJ Culpepper<br>
> Hamiltonian annealed importance sampling for partition function estimation<br>
> Redwood Technical Report (2011)
> http://arxiv.org/abs/1205.1925

The code in this repository can be used for log
likelihood estimation, partition function estimation, and importance
weight estimation.  It can also be used as a Hamiltonian Monte Carlo sampler.  See **HAIS_examples.m** for usage examples.

## Files
- **HAIS_examples.m** demonstrates the capabilities of this code in a variety of scenarios.
- **HAIS.m** performs Hamiltonian Annealed Importance Sampling.
- **HAIS_logL.m** calculates the log likelihood of a model given data using HAIS.
- **HAIS_logL_aux.m** calculates the log likelihood of a model with hidden (auxiliary) variables given data using HAIS.
