import h5py
import numpy as np
import matplotlib.pyplot as plt
from spod import spod
from utils import trapzWeightsPolar

plt.close('all')

data = {}
f = h5py.File('jet_data/jetLES.mat','r')
fields = ['p','x','r','dt']
for key, value in f.items():
    if key in fields:
        data[key] = np.array(value).astype(np.float64)

p = np.swapaxes(data['p'],0,2)
x = np.swapaxes(data['x'],0,1)
r = np.swapaxes(data['r'],0,1)
dt = data['dt'][0][0]

intWeights = trapzWeightsPolar(r[:,0], x[0,:])

result = spod(p, window=np.ones(256), weight=intWeights, noverlap=50, dt=dt, conflvl=0.99, debug=2)

plt.figure()
for mi in range(0, result['L'].shape[1], 5):
    line = plt.loglog(result['f'], result['L'][:,mi], linewidth=0.5)
    color = line[-1].get_color()
    plt.loglog(result['f'], result['Lc'][:,mi,0], '--',color=color, linewidth=0.5) # lower confidence level
    plt.loglog(result['f'], result['Lc'][:,mi,1], '--',color=color, linewidth=0.5) # upper confidence level

plt.xlabel('frequency')
plt.ylabel('SPOD mode energy')
plt.show()