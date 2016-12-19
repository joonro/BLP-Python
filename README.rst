==========
BLP-Python
==========

    :Author: Joon Ro
    :Contact: joon.ro@outlook.com
    :Date: 2016-12-16 FRI

Introduction
------------

``BLP-Python`` provides a Python implementation of random coefficient logit model
of Berry, Levinsohn and Pakes (1995).

Based on code for Nevo (2000b) at
`http://faculty.wcas.northwestern.edu/~ane686/supplements/rc_dc_code.htm <http://faculty.wcas.northwestern.edu/~ane686/supplements/rc_dc_code.htm>`_.

This code calculates analytical gradients and use tight tolerances for the
contraction mapping (Dube et al. 2012). With BFGS method, it quickly converges
to the optimum. (See Nevo (2000b) Example)

Notes on the code
~~~~~~~~~~~~~~~~~

- Use global states only for read-only variables

- Avoid inverting matrices whenever possible for numerical stability

- Use tighter tolerance for the contraction mapping

- Use greek unicode symbols whenever possible for readability

- Fix a small bug in the original code that prints wrong standard error twice
  for the mean estimates

- Market share integration is done in C via Cython, and it is parallelized
  across the simulation draws via openMP

Installation
------------

Dependencies
~~~~~~~~~~~~

- Python 3.5 (for ``@`` operator and unicode variable names). I recommend
  Anaconda Python Distribution, which comes with many of the scientific libraries,
  as well as ``conda``, a convenient script to install many packages.

- ``numpy`` and ``scipy`` for array operations and linear algebra

- ``cython`` for parallelized market share integration

- ``pandas`` for result printing

Download
~~~~~~~~

- With git:

  .. code-block:: sh
      :number-lines: 0

      git clone https://github.com/joonro/BLP-Python.git

- Or you can download the `master branch <https://github.com/joonro/BLP-Python/archive/master.zip>`_ as a zip archive

Compiling the Cython Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- I include the compiled Cython module (``_BLP.cp35-win_amd64.pyd``) for Python
  3.5 64bit, so you should be able to run the code without compiling the
  module in Windows. You have to compile it if you want to change the Cython
  module or if you are on GNU/Linux or Mac OS. GNU/Linux distributions come with
  ``gcc`` so it should be straightforward to compile the module.

- ``cd`` into the ``BLP-Python`` directory, and compile the cython module with
  the following command:

  .. code-block:: sh
      :number-lines: 0

      python setup.py build_ext --inplace

Windows
^^^^^^^

- For Windows users, to compile the cython module with the openMP
  (parallelization) support with 64-bit Python, you have to install Microsoft
  Visual C++ compiler following instructions at
  `https://wiki.python.org/moin/WindowsCompilers <https://wiki.python.org/moin/WindowsCompilers>`_. For Python 3.5, you either
  install Microsoft Visual C++ 14.0 standalone, or you can install Visual
  Studio 2015 which contains Visual C++ 14.0 compiler.

Nevo (2000b) Example
--------------------

``tests/test_replicate_Nevo_2000b.py`` replicates the results from Nevo
(2000b). In the main folder, you can run the script as:

.. code-block:: sh
    :number-lines: 0

    python "tests/test_replicate_Nevo_2000b.py"

It evaluates the objective function at the starting values and creates the
following results table:

.. code-block:: python
    :number-lines: 0

                   Mean        SD      Income  Income^2       Age     Child
    Constant  -1.833294  0.377200    3.088800  0.000000  1.185900   0.00000
               0.257829  0.129433    1.212647  0.000000  1.012354   0.00000
    Price    -32.446922  1.848000   16.598000 -0.659000  0.000000  11.62450
               7.751913  1.078371  172.776110  8.979257  0.000000   5.20593
    Sugar      0.142915 -0.003500   -0.192500  0.000000  0.029600   0.00000
               0.012877  0.012297    0.045528  0.000000  0.036563   0.00000
    Mushy      0.801608  0.081000    1.468400  0.000000 -1.514300   0.00000
               0.203454  0.206025    0.697863  0.000000  1.098321   0.00000
    GMM objective: 14.900789417017275
    Min-Dist R-squared: 0.2718388379589566
    Min-Dist weighted R-squared: 0.0946528053333926

Note that standard errors are slightly different because I avoid inverting
matrices as much as possible in calculations. In addition, the original code
has a minor bug in the standard error printing. That is, in ``rc_dc.m``, line
102, ``semcoef = [semd(1); se(1); semd];`` should be ``semcoef = [semd(1); se(1); semd(2:3)];`` instead (``0.258`` is printed twice as a result).

In addition, this code uses tighter tolerance for the contraction mapping, and
with the simplex (``Nelder-Mead``) optimization method, this code minimizes the
GMM objective function to the correct minimum of ``4.56``.

After running the code, you can try the full estimation with:

.. code-block:: python
    :number-lines: 0

    BLP.estimate(θ20=θ20)

For example, in a IPython console:

.. code-block:: python
    :number-lines: 0

    %run "tests/test_replicate_Nevo_2000b.py"
    BLP.estimate(θ20=θ20)

You should get the following results:

.. code-block:: python
    :number-lines: 0

    Optimization terminated successfully.
             Current function value: 4.561515
             Iterations: 45
             Function evaluations: 50
             Gradient evaluations: 50

                   Mean        SD      Income   Income^2       Age      Child
    Constant  -2.009919  0.558094    2.291972   0.000000  1.284432   0.000000
               0.326997  0.162533    1.208569   0.000000  0.631215   0.000000
    Price    -62.729902  3.312489  588.325237 -30.192021  0.000000  11.054627
              14.803215  1.340183  270.441021  14.101230  0.000000   4.122563
    Sugar      0.116257 -0.005784   -0.384954   0.000000  0.052234   0.000000
               0.016036  0.013505    0.121458   0.000000  0.025985   0.000000
    Mushy      0.499373  0.093414    0.748372   0.000000 -1.353393   0.000000
               0.198582  0.185433    0.802108   0.000000  0.667108   0.000000
    GMM objective: 4.5615146550344186
    Min-Dist R-squared: 0.4591043336106454
    Min-Dist weighted R-squared: 0.10116438381046189

You can check the gradient at the optimum:

.. code-block:: python
    :number-lines: 0

    >>> BLP.gradient_GMM(BLP.results['θ2']['x'])
    contraction mapping finished in 0 iterations

    array([  1.23888940e-07,   1.15056001e-08,   1.58824491e-08,
            -4.45649242e-08,  -9.61452074e-08,  -1.75233503e-08,
            -9.94539619e-07,   9.60900497e-08,  -3.30553299e-07,
             1.24174991e-07,   4.17569410e-07,   1.33642515e-07,
             1.94273594e-09])

I verified that the optimum is achieved with ``Nelder-Mead`` (simplex), ``BFGS``,
``TNC``, and ``SLSQP`` ```scipy.optimize`` <https://www.docs.scipy.org/doc/scipy/reference/optimize.html>`_ methods. ``BFGS`` and ``SLSQP`` were the
fastest.

Unit Testing
------------

I use ``pytest`` for unit testing. You can run them with:

.. code-block:: python
    :number-lines: 0

    python -m pytest

References
----------

Berry, S., Levinsohn, J., & Pakes, A. (1995). *Automobile Prices In Market Equilibrium*. Econometrica, 63(4), 841.

Dubé, J., Fox, J. T., & Su, C. (2012). Improving the Numerical Performance of
BLP Static and Dynamic Discrete Choice Random Coefficients Demand
Estimation. Econometrica, 1–34.

Nevo, A. (2000). *A Practitioner’s Guide to Estimation of Random-Coefficients Logit Models of Demand*. Journal of Economics & Management Strategy, 9(4),
513–548.

Changelog
---------

0.4.0
~~~~~

- Implement estimation of parameter means

- Implement standard error calculation

- Add Nevo (2000b) example

- Add a unit test

- Improve README

0.3.0
~~~~~

- Implement GMM objective function and estimation of :math:`\theta_{2}`
