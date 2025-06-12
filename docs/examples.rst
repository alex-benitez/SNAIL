Examples
========
This is a suite of examples to give you an idea of how the code works; all of the code here can be found in the /examples directory.

1D Single Atom Response
-----------------------
The first step when using SNAIL is always to define the config, this stores all of the information about your laser and target and allows the user to only define essential variables. 
.. code-block:: python

 	pulse = generate_pulse("gaussian")

Cosine-Squared

.. code-block:: python

 	pulse = generate_pulse("cos_sqr")
