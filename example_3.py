#  EXAMPLE 3: Manually specify spectral estimation parameters and use cell volume weighted inner product.
#  The large-eddy simulation data provided along with this example is a
#  subset of the database of a Mach 0.9 turbulent jet described in [1] and
#  was calculated using the unstructured flow solver Charles developed at
#  Cascade Technologies. If you are using the database in your research or
#  teaching, please include explicit mention of Brès et al. [1]. The test
#  database consists of 5000 snapshots of the symmetric component (m=0) of
#  a round turbulent jet. A physical interpretaion of the SPOD results is
#  given in [2], and a comprehensive discussion and derivation of SPOD and
#  many of its properties can be found in [3].
#
#   References:
#     [1] G. A. Brès, P. Jordan, M. Le Rallic, V. Jaunet, A. V. G.
#         Cavalieri, A. Towne, S. K. Lele, T. Colonius, O. T. Schmidt,
#         Importance of the nozzle-exit boundary-layer state in subsonic
#         turbulent jets, J. of Fluid Mech. 851, 83-124, 2018
#     [2] Schmidt, O. T. and Towne, A. and Rigas, G. and Colonius, T. and
#         Bres, G. A., Spectral analysis of jet turbulence, J. of Fluid Mech. 855, 953–982, 2018
#     [3] Towne, A. and Schmidt, O. T. and Colonius, T., Spectral proper
#         orthogonal decomposition and its relationship to dynamic mode
#         decomposition and resolvent analysis, J. of Fluid Mech. 847, 821–867, 2018

import h5py
import numpy as np
import matplotlib.pyplot as plt
from spod import spod
from utils import trapzWeightsPolar

plt.close("all")

data = {}

f = h5py.File("jet_data/jetLES.mat", "r")
p = np.swapaxes(f["p"], 0, 2)
x = np.swapaxes(f["x"], 0, 1)
r = np.swapaxes(f["r"], 0, 1)
dt = f["dt"][0][0]

# SPOD of the test database.
#   In this example, we manually specify a rectangular window of length 256
#   and an overlap of 50 snaphots. Furthermore, we use trapezoidal
#   quadrature weights to define a physical inner product corresponding to
#   the volume integral over the pressure squared.

# trapezoidal quadrature weights for cylindrical coordinates
intWeights = trapzWeightsPolar(r[:, 0], x[0, :])

# SPOD using a rectangular window of length 256 and 50 snaphots overlap
result = spod(p, window=np.ones(256), weight=intWeights, noverlap=50, dt=dt, debug=2)

# Plot the SPOD spectrum and some modes as before.
plt.figure()
plt.loglog(result["f"], result["L"])
plt.xlabel("frequency")
plt.ylabel("SPOD mode energy")
plt.show()

plt.figure(figsize=(8, 5))
count = 1
for fi in [10, 15, 25]:
    for mi in range(2):
        plt.subplot(3, 2, count)
        P = np.real(result["P"][fi - 1, :, :, mi])
        vmax = np.max(np.abs(P))
        plt.contourf(x, r, P, 256, vmin=-vmax, vmax=vmax)
        plt.axis("scaled")
        plt.xlabel("x")
        plt.ylabel("r")
        plt.title(
            "f=%.2f, mode %d, $\lambda$ = %.2g"
            % (result["f"][fi - 1], mi + 1, result["L"][fi - 1, mi])
        )
        plt.xlim(0, 10)
        plt.ylim(0, 2)
        count += 1

plt.show()
