
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10, 10 , 100)
import math 
e = math.e
pi = math.pi
a = 3.0
b = 2.0
c = -0.5*(((x-a)/b)**2)
fx = 1 / (b * (2*pi**0.5)) * e**(c)

plt.figure(figsize=(10, 6))
plt.plot(x, fx, "-*",color='blue')

plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)

plt.show()
