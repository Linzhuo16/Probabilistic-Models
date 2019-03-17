import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)
N = 1000000

x = np.random.uniform(0, 1, N)
y = np.sqrt(x)

x1 = np.linspace(0,1,100)
y1 = 2 * x1

hist, bins = np.histogram(y, bins=40) # Calculate histogram
plt.cla()
plt.title('N=' + str(N))
plt.xlabel('u')
plt.ylabel('p(u)')


plt.hist(y, bins=bins, density=True) # normalize
plt.plot(x1, y1, c='r')
plt.show()
