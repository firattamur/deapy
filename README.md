<h3 align="center">
	dea-py
</h3>

<!-- badges -->
<p align="center">

<!-- language -->
<img src="https://img.shields.io/badge/Python-3.8.1-success" alt="Python: 3.8.1">
<img src="https://img.shields.io/badge/Pyomo-6.1.2-yellow" alt="Pyomo: 6.1.2">
<img src="https://img.shields.io/badge/-GLPK-blue" alt="GLPK">
<img src="https://img.shields.io/badge/-IPOPT-blue" alt="Ipopt">
  
  
<!-- inprogress or completed -->
<!-- <img src="https://img.shields.io/badge/-completed-green" alt="completed"> -->
	
<!-- inprogress or completed -->
<img src="https://img.shields.io/badge/-in%20progress-red" alt="in progress">
	
<!-- licence -->
<img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License: MIT">
	
<!-- week of year -->
<!-- <img src="https://img.shields.io/badge/week-30-green" alt="in progress"> -->

</p>

![DataEnvelopmentAnalysis logo](assets/logo/dea-py-logo.png "DataEnvelopmentAnalysis logo")

<!-- | Documentation | Build Status      | Coverage    | Zenodo      |
|:-------------:|:-----------------:|:-----------:|:-----------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] |  [![][githubci-img]][githubci-url] | [![][codecov-img]][codecov-url] | [![][zenodo-img]][zenodo-url] | -->

<hr>

A Python wrapper for [DataEnvelopmentAnalysis.jl](https://github.com/javierbarbero/DataEnvelopmentAnalysis.jl) library.

* Python `3.8.1` and above on Linux, macOS, and Windows

* [Pyomo](https://github.com/Pyomo/pyomo) LP modeling library

* [GLPK](http://www.gnu.org/software/glpk/) and [IPOPT](https://coin-or.github.io/Ipopt/) LP solvers

<hr>

## Installation

The package can be installed with the Python Package Index (PyPI):
```python
pip install dea-py
```

<hr>

### Technical Efficiency DEA Models:


- [X] Radial input and output oriented model
- [ ] Directional distance function model
- [X] Additive models: 
	* Weighted Additive Model
 	* Measure of Inefficiency Proportions Model(MIP)
 	* Normalized Weighted Additive Model
 	* Range Adjusted Measure Model(RAM)
 	* Bounded Adjusted Measure Model (BAM)
- [ ] Generalized distance function model
- [ ] Russell graph and oriented model
- [ ] Enhanced Russell Graph Slack Based Measure
- [ ] Modified directional distance function
- [ ] Hölder distance function
- [ ] Reverse directional distance function

<hr>

### Economic Efficiency DEA Models:


- [ ] Cost model
- [ ] Revenue model
- [ ] Profit model
- [ ] Profitability model

<hr>

### Productivity Change DEA Models:


- [ ] Mamlmquist index.

<hr>

## Authors

DataEnvelopmentAnalysis.py is being developed by [firattamur](https://github.com/firattamur)
