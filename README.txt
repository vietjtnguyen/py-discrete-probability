=====================
pyDiscreteProbability
=====================

Description
-----------

pyDiscreteProbability provides a number of classes and functions for dealing
with discrete probabilities. These include marginal probability tables
(``MarginalTable``), conditional probability tables (``ConditionalTable``), and
Bayesian networks.

This module is not intended for any production and/or scientific use. It was
developed for my own learning purposes thus serves a primarily education
purpose. The demands of production or scientific use, such as good numerical
conditioning and good verification and validation coverage are not guaranteed.

* `Python Package Index Page <https://pypi.python.org/pypi/pyDiscreteProbability/>`_
* `Github Repository <https://github.com/vietjtnguyen/py-discrete-probability>`_

Examples
--------

A simple example can be found at http://nbviewer.ipython.org/5883568.

References
----------

I am current using `Adnan Darwiche's "Modeling and Reasoning with Bayesian Networks" <http://www.amazon.com/Modeling-Reasoning-Bayesian-Networks-Darwiche/dp/0521884381/ref=sr_1_1>`_ as reference. It is an excellent text.

For an actual production/scientific level implementation look at `SamIam <http://reasoning.cs.ucla.edu/samiam>`_ from `Adnan Darwiche's Automated Reasoning <http://reasoning.cs.ucla.edu/>`_ group. I am not affiliated with this group.

TODO
----

I doubt I'll ever get to all of this but here's a list of things that can be done. Most of it is based off of "Modeling and Reasoning with Bayesian Networks" by Adnan Darwiche.

* Reorganizing class hierarchy so that a ``MarginalTable`` is derived from ``ConditionalTable``
* Multiplying together ``MarginalTable``s
* Multiplying together ``MarginalTable``s and ``ConditionalTable``s
* Expanded unit tests
* Expectation maximization learning for ``MarginalTable``
* Expectation maximization learning for ``BayesianNetwork``
* Function-call interface for querying ``BayesianNetwork``s
  * Start off accepting only full marginal queries, other queries can be answered by converting to ``MarginalTable``
  * Later inference methods: variable elimination, factor elimination (jointree), network polynomial, belief propagation, model counting
* Approximate inference for ``BayesianNetwork``s using stochastic sampling techniques
  * Direct sampling
  * Importance sampling
  * Particle filtering
  * Markov chain Monte Carlo
  * Gibbs sampling
* Approximate inference for ``BayesianNetwork``s using belief propagation
* Algorithms for detecting d-separation and answer conditional independence queries
  * Maybe automated reasoning to see if a conditional independence is implied by a starting set of conditional independencies and the graphoid axioms
* Maximum a posterior (MAP) queries
* Most probable explanation (MPE) queries
* Methods for performing do-interventions
* Bayesian learning for ``BayesianNetwork`` using meta-networks and Dirichlet priors for continuous parameter sets
* Network structure learning for ``BayesianNetwork`` using max-likelihood approach and various regularization techniques (AIC, BIC/MDL).
* Network structure learning for ``BayesianNetwork`` using greedy local search
* Network structure learning for ``BayesianNetwork`` using constraint based approach (e.g. IC, IC*, PC, etc.)
* Network structure learning for ``BayesianNetwork`` using Bayesian approach and Bayesian Dirichlet (BD) score
* Test on http://www.cs.huji.ac.il/~galel/Repository/ and/or http://www.norsys.com/netlibrary/index.htm
* Comment code
* Produce documentation using Sphinx

