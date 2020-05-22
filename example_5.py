import h5py
import numpy as np
import matplotlib.pyplot as plt
from spod import spod
from utils import trapzWeightsPolar, getjet

plt.close('all')

data = {}

f = h5py.File('jet_data/jetLES.mat','r')
fields = ['x','r','dt','p_mean']
for key, value in f.items():
    if key in fields:
        data[key] = np.array(value).astype(np.float64)

p_mean = np.swapaxes(data['p_mean'],0,1)
x = np.swapaxes(data['x'],0,1)
r = np.swapaxes(data['r'],0,1)
dt = data['dt'][0][0]

intWeights = trapzWeightsPolar(r[:,0], x[0,:])

result = spod(getjet, window=128, weight=intWeights, noverlap=64, dt=dt, mean=p_mean, nt=2000, debug=2)

plt.figure()
plt.loglog(result['f'], result['L'], linewidth=0.5)
plt.xlabel('frequency')
plt.ylabel('SPOD mode energy')
plt.show()