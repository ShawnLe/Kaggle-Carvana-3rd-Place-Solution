
import utils.island_counter as isl 
import numpy as np
import matplotlib.pyplot as plt

map = np.random.randint(0,2,size=(10,10))
isl_map = isl.count(map)

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(map)
fig.add_subplot(1,2,2)
plt.imshow(isl_map)
plt.show()




