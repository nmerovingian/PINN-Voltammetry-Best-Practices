import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

Time  = np.linspace(0.05,0.9,num=200)

Analytical_J = 0.7854+0.4431/np.sqrt(Time)+0.2146*np.exp(0.39115/np.sqrt(Time))

plt.plot(Time,Analytical_J)

plt.show()


