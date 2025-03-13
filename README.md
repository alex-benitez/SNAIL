# SNAIL
SFA Numerical Accelerated Integration Library (SNAIL) is a package that allows users to generate the dipole response of gas particles in strong fields, using the Lewenstein approximation for SFA.
This code was inspired by HHGMax, and follows the same algorithm. The main integral adapted from the original Lewenstein paper:

![Equation](https://latex.codecogs.com/svg.image?\inline&space;{\color{white}d(t)=-i\cdot&space;e_x\int_{0}^{\tau_{max}}d\tau\;\omega(\tau)\cdot\left(\frac{\pi}{\epsilon&plus;i\tau/2}\right)^{3/2}\cdot[\mathbf{E}(t)\cdot\mathbf{D}(\mathbf{p}_s(t,\tau)-\mathbf{A}(t-\tau))]\times&space;exp(-iS_s(t,\tau))\cdot\mathbf{D^*}(\mathbf{p}_s(t,\tau)-\mathbf{A}(t))&plus;c.c.})

Which calculates the dipole response of an electron ionized into the continuum by a strong laser field with no ground state depletion. For more information check out the docs and examples (work in progress).


