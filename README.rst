.. highlight:: sh

==============
 Introduction
==============

:Date: November 28, 2014
:Version: 0.3.0
:Authors: Joon H. Ro, joon.ro[at]outlook.com
:Web site: https://github.com/joonro/BLP-Python
:Copyright: This document has been placed in the public domain.
:License: BLP-Python is released under the GPLv3.


Purpose
=======

BLP-Python provides a Python implementation of random coefficient logit model
of Berry, Levinsohn and Pakes (1995).


Installation
============

Dependencies
------------

* `The SciPy Stack <http://www.scipy.org/stackspec.html>`_

Download
--------

* With git::

   git clone https://github.com/joonro/BLP-Python.git

* Or download the master branch as a `zip archive
  <https://github.com/joonro/BLP-Python/archive/master.zip>`_


Compiling Cython Module
-----------------------

* ``cd`` into the `BLP-Python` directory, and compile the cython module with
  the following command:

.. code-block:: sh

    python setup.py build_ext --inplace

References
==========

Berry, Steven, James Levinsohn, and Ariel Pakes. "Automobile prices in market
equilibrium." Econometrica (1995).
