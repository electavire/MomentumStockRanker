import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score

X = np.arange(1, 10, 1)
Y = np.power(2,X)
print(Y)

#Take the log of the Y values to 
lnY = np.log(Y)

plt.scatter(X, Y)
plt.show()

regression = linear_model.LinearRegression()

print(r2_score(X,[5,5,5,5,5,5,5,5,5]))

