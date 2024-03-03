import matplotlib.pyplot as plt
import numpy as np

track = np.zeros((100, 100))
track[30:, 60:] = -np.inf

# for i in range(15):
#     track[i:, i+3:] = np.inf

plt.imshow(track, cmap='gray')
plt.show()
