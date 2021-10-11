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

![DataEnvelopmentAnalysis logo](assets/logo/logo.png "DataEnvelopmentAnalysis logo")

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

```python
radio_dea = RadialDEA()

X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])
Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])

radio_dea = RadialDEA(orient=Orient.Input, rts=RTS.CSR, disposX=Dispos.Strong, disposY=Dispos.Strong)
radio_dea.fit(X, Y)
radio_dea.dea()
radio_dea.pprint()

```

- [X] Additive Models: 

	```python
	X = np.array([[5, 13], [16, 12], [16, 26], [17, 15], [18, 14], [23, 6], [25, 10], [27, 22], [37, 14], [42, 25], [5, 17]])
	Y = np.array([[12], [14], [25], [26], [8], [9], [27], [30], [31], [26], [12]])
	```

	* Weighted Additive Model

	```python
	    additive_dea = AdditiveDEA()
	```

 	* Measure of Inefficiency Proportions Model(MIP)

	```python
	    additive_dea = AdditiveDEA(model=AdditiveModels.MIP)
	```

 	* Normalized Weighted Additive Model (NORM)

	```python
	    additive_dea = AdditiveDEA(model=AdditiveModels.NORM)
	```

 	* Range Adjusted Measure Model (RAM)

	```python
	    additive_dea = AdditiveDEA(model=AdditiveModels.RAM)
	```

 	* Bounded Adjusted Measure Model (BAM)

	```python
	    additive_dea = AdditiveDEA(model=AdditiveModels.BAM)
	```
	
	```python
	 additive_dea.fit(X, Y)
	 additive_dea.dea()
	 additive_dea.pprint()
	```

<hr>


## Authors

DataEnvelopmentAnalysis.py is being developed by [firattamur](https://github.com/firattamur)
