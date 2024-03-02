from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

independent_vars = []
dependent_vars = []
range_vars = [] #index 0-3: min-max X, min-max y

def input_vars(list_name):
    n = int(input("Nhập số lượng phần tử: "))

    for i in range(n):
        element = float(input("Nhập phần tử {}: ".format(i+1)))
        list_name.append(element)

input_vars(independent_vars)
input_vars(dependent_vars)
input_vars(range_vars)

# Name x,y label
x_label = ""
y_label = ""

def input_name(prompt):
    return input(prompt)

x_label = input_name("Nhập tên cho trục x: ")
y_label = input_name("Nhập tên cho trục y: ")

# independent_vars (unit)
X = np.array([independent_vars]).T
# dependent_vars (unit)
y = np.array([dependent_vars]).T
# Visualize data 
plt.plot(X, y, 'ro')
plt.axis(range_vars)
plt.xlabel(x_label)
plt.ylabel(y_label)
#plt.show()

# Building Xbar 
one = np.ones((X.shape[0], 1)) #create a matrix with number row of X and 1 column
Xbar = np.concatenate((one, X), axis = 1) #concatenate one and X

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar) #A = Xbar^T * Xbar
b = np.dot(Xbar.T, y) #b = Xbar^T * y
w = np.dot(np.linalg.pinv(A), b) #w = A^-1 * b
print('w = ', w)

# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(range_vars[0] + 5, range_vars [1] - 5) #create a list from 145 to 185 with 2 elements (x0, x1)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis(range_vars)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.show()
