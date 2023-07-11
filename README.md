# UBayFSpy <img src="logo.png" align="right" width="200"/>


The UBayFS package implements the framework proposed in the article [Jenul et al. (2022)](https://link.springer.com/article/10.1007/s10994-022-06221-9). UBayFS is an ensemble feature selection technique embedded in a Bayesian statistical framework. The method combines data and user knowledge, where the first is extracted via data-driven ensemble feature selection. The user can control the feature selection by assigning prior weights to features and penalizing specific feature combinations. In particular, the user can define a maximal number of selected features and must-link constraints (features must be selected together) or cannot-link constraints (features must not be selected together). Using relaxed constraints, a parameter $\rho$ regulates the penalty shape. Hence, violation of constraints can be valid but leads to a lower target value of the feature set that is derived from the violated constraints. UBayFS can be used for common feature selection and also for block feature selection.

Documentation
-------------
A [documentation](https://annajenul.github.io/UBayFSpy/) illustrates how UBayFS can be used for standard feature selection 


Requirements and Dependencies
-----------------------------

- numpy>=1.23.5
- pandas>=1.5.3
- scikit-learn>=1.2.2
- scipy>=1.10.0
- random
- mrmr>=0.2.6
- pygad>=3.0.1
- math


Implementation Details
----------------------
The original paper defines the following utility function $U(\boldsymbol{\delta},\boldsymbol{\theta})$ for optimization with respect to $\boldsymbol{\delta}\in \lbrace 0,1\rbrace ^N$:
$$U(\boldsymbol{\delta},\boldsymbol{\theta}) = \boldsymbol{\delta}^T \boldsymbol{\theta}-\lambda \kappa(\boldsymbol{\delta}), $$
for fixed $\lambda>0$.


For practical reasons, the implementation in the UBayFS package uses a modified utility function $\tilde{U}(\boldsymbol{\delta},\boldsymbol{\theta})$ which adds an admissibility term $1-\kappa(\boldsymbol{\delta})$ rather than subtracting an inadmissibility term $\kappa(\boldsymbol{\delta})$
$$\tilde{U}(\boldsymbol{\delta},\boldsymbol{\theta}) = \boldsymbol{\delta}^T \boldsymbol{\theta}+\lambda (1-\kappa(\boldsymbol{\delta})) = \boldsymbol{\delta}^T \boldsymbol{\theta}-\lambda \kappa(\boldsymbol{\delta}) +\lambda.$$

Thus, the function values of $U(\boldsymbol{\delta},\boldsymbol{\theta})$ and $\tilde{U}(\boldsymbol{\delta},\boldsymbol{\theta})$ deviate by a constant $\lambda$; however, the optimal feature set $$\boldsymbol{\delta}^{\star} = \underset{\boldsymbol{\delta}\in\lbrace 0,1\rbrace ^N}{\text{arg max}}~ U(\boldsymbol{\delta},\boldsymbol{\theta}) = \underset{\boldsymbol{\delta}\in\lbrace 0,1\rbrace ^N}{\text{arg max}}~ \tilde{U}(\boldsymbol{\delta},\boldsymbol{\theta})$$ remains unaffected.


Installation
------------
To install the package with the pip package manager, run the following command:  
`python3 -m pip install git+https://github.com/annajenul/UBayFSpy.git`

Contributing
------------
Your contribution to UBayFS is very welcome! 

Contribution to the package requires the agreement of the [Contributor Code of Conduct](https://github.com/annajenul/UBayFSpy/blob/main/CODE_OF_CONDUCT.md) terms.

For the implementation of a new feature or bug-fixing, we encourage you to send a Pull Request to [the repository](https://github.com/annajenul/UBayFSpy). Please add a detailed and concise description of the invented feature or the bug. In case of fixing a bug, include comments about your solution. To improve UBayFS even more, feel free to send us issues with bugs, you are not sure about. We are thankful for any kind of constructive criticism and suggestions.

Citation
------------
If you use UBayFS in a report or scientific publication, we would appreciate citations to the following paper
Jenul, A., Schrunner, S. et al. A user-guided Bayesian framework for ensemble feature selection in life science applications (UBayFS). Mach Learn (2022). https://doi.org/10.1007/s10994-022-06221-9

Bibtex entry:

	@article{Jenul2022,
	  doi = {10.1007/s10994-022-06221-9},
	  url = {https://doi.org/10.1007/s10994-022-06221-9},
	  year = {2022},
	  month = aug,
	  publisher = {Springer Science and Business Media {LLC}},
	  volume = {111},
	  number = {10},
	  pages = {3897--3923},
	  author = {Anna Jenul and Stefan Schrunner and J\"{u}rgen Pilz and Oliver Tomic},
	  title = {A user-guided Bayesian framework for ensemble feature selection in life science applications ({UBayFS})},
	  journal = {Machine Learning}
}
