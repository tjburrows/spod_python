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

result = spod(p, window=np.ones(256), weight=intWeights, noverlap=50, dt=dt, debug=2)

plt.figure()
plt.loglog(result['f'], result['L'], linewidth=0.5)
plt.xlabel('frequency')
plt.ylabel('SPOD mode energy')
plt.show()

plt.figure(figsize=(8,5))
count = 1
for fi in [10,15,25]:
    for mi in range(2):
        plt.subplot(3,2,count)
        P = np.real(result['P'][fi-1,:,:,mi])
        vmax = np.max(np.abs(P))
        plt.contourf(x, r, P, 256, vmin=-vmax, vmax=vmax)
        plt.axis('scaled')
        plt.xlabel('x')
        plt.ylabel('r')
        plt.title('f=%.2f, mode %d, $\lambda$ = %.2g' % (result['f'][fi-1], mi+1, result['L'][fi-1,mi]))
        plt.xlim(0,10)
        plt.ylim(0,2)
        count += 1
        
plt.show()