---
title: 'BLP-Python: Random Coefficient Logit Model in Python'
tags:
  - blp
  - python
  - economics
  - marketing
  - industrial organization
  - econometrics
  - structural model
authors:
 - name: Joon H. Ro
   orcid: 0000-0002-5895-163X
   affiliation: 1
affiliations:
 - name: Tulane University
   index: 1
date: 22 September 2017
bibliography: paper.bib
nocite: |
  @BLP_Python_GitHub
---

# Summary

BLP-Python provides a Python implementation of random coefficient logit model
of [@Berry_Levinsohn_Pakes_1995_Econometrica] (henceforth BLP), which is
widely used in Economics (e.g., [@Nevo_2001_Econometrica];
[@Petrin_2002_J_Polit_Economy]) and Marketing (e.g.,
[@Sudhir_2001_Market_Sci]) for demand estimation from aggregate data. The
specific implementation follows the model described in
[@Nevo_2000_J_Econ_Manage_Strategy].

A user should provide data for estimation and random draws for simulating
integrals as multidimensional `xarray.DataArray` objects. For better
performance, calculations of mean utilities and individual choice
probabilities are implemented with Cython with parallel loop via openMP.

This package should be useful for researchers who want to estimate BLP-related
model. Depending on the user's model specification (such as different utility
specification, adding the supply side, and/or using micro-moments),
modification of the code will be required. Hence, the code is written with
readability in mind. For example, greek letters are used for variable names
whenever possible to help understand the code.

# References
  
