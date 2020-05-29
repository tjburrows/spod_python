import numpy as np
import h5py


def getjet(i):
    f = h5py.File("jet_data/jetLES.mat", "r")
    return f["p"][:, :, i].T


def trapzWeightsPolar(r, z):
    """ Integration weight matrix for cylindical coordinates using trapazoidal rule """

    weight_thetar = np.zeros(r.shape[0])
    weight_thetar[0] = 0.25 * np.pi * (r[0] + r[1]) ** 2
    weight_thetar[1:-1] = (
        0.25 * np.pi * ((r[1:-1] + r[2:]) ** 2 - (r[1:-1] + r[:-2]) ** 2)
    )
    weight_thetar[-1] = np.pi * (r[-1] ** 2 - 0.25 * (r[-1] + r[-2]) ** 2)

    weight_z = np.gradient(z)
    weight_z[0] = 0.5 * (z[1] - z[0])
    weight_z[-1] = 0.5 * (z[-1] - z[-2])

    weight_rz = np.matmul(np.expand_dims(weight_thetar, 1), np.expand_dims(weight_z, 0))

    return weight_rz
