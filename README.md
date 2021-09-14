# DataEnvelopmentAnalysis.py

![DataEnvelopmentAnalysis logo](assets/logo/dea-py-logo.png "DataEnvelopmentAnalysis logo")

<!-- | Documentation | Build Status      | Coverage    | Zenodo      |
|:-------------:|:-----------------:|:-----------:|:-----------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] |  [![][githubci-img]][githubci-url] | [![][codecov-img]][codecov-url] | [![][zenodo-img]][zenodo-url] | -->

A Python wrapper for [DataEnvelopmentAnalysis.jl](https://github.com/javierbarbero/DataEnvelopmentAnalysis.jl) library.

* Python `3.8.1` and above on Linux, macOS, and Windows.

* [Pyomo](https://github.com/Pyomo/pyomo) lp modeling library

* [GLPK](http://www.gnu.org/software/glpk/) and [Ipopt](https://coin-or.github.io/Ipopt/) lp solvers

## Installation

The package can be installed with the Python Package Index (PyPI):
```python
pip install dea-py
```

## Available models

**Technical efficiency DEA models**

- [X] Radial input and output oriented model.
- [ ] Directional distance function model.
- [X] Additive models: weighted additive model, measure of inefficiency proportions (MIP), normalized weighted additive model, range adjusted measure (RAM), bounded adjusted measure (BAM).
- [ ] Generalized distance function model.
- [ ] Russell graph and oriented model.
- [ ] Enhanced Russell Graph Slack Based Measure.
- [ ] Modified directional distance function.
- [ ] Hölder distance function.
- [ ] Reverse directional distance function.

**Economic efficiency DEA models**

- [ ] Cost model.
- [ ] Revenue model.
- [ ] Profit model.
- [ ] Profitability model.

**Productivity change models**

- [ ] Mamlmquist index.

## Authors

DataEnvelopmentAnalysis.py is being developed by [Firat Tamur](https://github.com/firattamur)
