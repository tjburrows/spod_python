# -*- coding: utf-8 -*-

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from spod import spod
plt.close('all')
data = {}

f = h5py.File('jet_data/jetLES.mat','r')
for key, value in f.items():
    data[key] = np.array(value)

p = np.moveaxis(data['p'],2,0)
x = data['x']
r = data['r']

# print(p.shape)

fig1 = plt.figure()
plt.xlabel('x')
plt.ylabel('r')
plt.axis('equal')
cax = plt.pcolormesh(x, r, p[0,:,:], shading='gouraud', cmap='viridis', vmin=4.43,vmax=4.48)

def animate(i):
    cax.set_array(p[i,:,:].flatten())

ani = animation.FuncAnimation(fig1, animate, interval=50, frames=100, repeat=False, blit=False)
plt.show()

result = spod(p, debug=2)

plt.figure()
plt.loglog(result['L'], linewidth=0.5)
plt.xlabel('frequency index')
plt.ylabel('SPOD mode energy')
plt.show()