# #### ex 1:
# ##### 对 Runge 函数：
# $$
# R(x) = \frac{1}{1+x^2},\quad x\in[-5,5]
# $$
# ##### 利用下列条件做插值、逼近并与 $R(x)$ 的图像进行比较
# ##### (1) 用等距节点 $x_i = -5 + i \; (i = 0, 1, \cdots, 10)$, 绘出它的 10 次  Newton 插值多项式的图像;
# ##### (2) 用节点 $x_i = 5 \cos \left ( \frac{2i+1}{42} \pi \right ) \; (i = 0, 1, \cdots, 20)$, 绘出它的 20 次 Lagrange 插值多项式的图像;
# ##### (3) 用等距节点 $x_i = -5 + i \; (i = 0, 1, \cdots, 10)$, 绘出它的分段线性插值函数的图像.

# ##### Newton interpolation 
# $$
# f(x) \approx N_n(x) = f[x_0] + f[x_0, x_1] (x - x_0) + \cdots + f[x_0, x_1, \cdots, x_{n-1}] (x - x_0)(x - x_1) \cdots (x - x_{n-1}).
# $$
# 
# ##### *Remark*:
# $$
# f[x_0, x_1, \cdots, x_k] = \sum_{j = 0}^k \frac{f(x_j)}{(x_j - x_0)(x_j - x_1) \cdots (x_j - x_{j-1})(x_j - x_{j+1}) \cdots (x_j - x_k)}.
# $$
# 
# ##### Lagrange interpolation
# $$
# f(x) \approx L_n(x) = \sum_{i = 0}^n f(x_i) l_i(x) \quad \text{with} \quad l_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}
# $$

# # /*
# #  * @Author: Keke Wu
# #  * @Date: 2022-03-02 12:38:35
# #  * @Last Modified by:   Keke Wu
# #  * @Last Modified time: 2022-03-02 12:38:35
# #  */

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import os
from make_dir import mkdir
mkdir(file_dir="./figure")


def R(x):
    return 1.0 / (1.0 + x**2)


def newton_interp(grids, x):
    
    grid_x, grid_y = grids
    if len(grid_x) != len(grid_x):
        print("Error: Length of grid_x and grid_y must be the same !")
    
    else:
        newton_sum = 0.0
        for i in range(len(grid_x)):
            basis_func = 1.0
            for j in range(i):
                basis_func *= x - grid_x[j]
                
            coef = 0.0
            for j in range(i + 1):
                g = 1.0
                for k in range(i + 1):
                    if k != j:
                        dist = grid_x[j] - grid_x[k]
                        if abs(dist) > 1.0e-8:
                            g *= dist
                        else: # can be optimized
                            print("[Error: The distance between x_{:d} and x_{:d} are too small !]".format(k, j))
            
                coef += grid_y[j] / g

            newton_sum += coef * basis_func
            
        return newton_sum


x = np.linspace(-5, 5, 11)
# x = 5.0 * np.cos((2.0*x + 1.0) / 42.0 * np.pi)
xx = np.linspace(-5, 5, 101)
y = newton_interp(grids = [x, R(x)], x = xx)
plt.style.use("seaborn-dark") 
fig = plt.figure()
plt.plot(x, R(x), color = "r", linewidth = 0.0, marker = "x", label = "grids")
plt.plot(xx, y, color = "b", linewidth = 1.0, linestyle = 'dashed', label = r"$N_n(x) \; (n = 10)$")
plt.plot(xx, R(xx), color = "k", label = r"$R(x) = 1 / (1+x^2)$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid()
plt.legend()
plt.savefig("./figure/newton_interpolation.pdf")
plt.show()


def lagrange_interp(grids, x): 

    grid_x, grid_y = grids
    if len(grid_x) != len(grid_x):
        print("Error: Length of grid_x and grid_y must be the same !")
        
    else:
        lagrange_sum = 0.0
        length = len(grid_x)
        for i in range(length):
            x_i = grid_x[i]
            basis_func = 1.0
            for j in range(length):
                if (i != j):
                    dist = x_i - grid_x[j] 
                    if abs(dist) > 1.0e-8:
                        basis_func *= (x - grid_x[j]) / dist
                    else: # can be optimized
                        print("[Error: The distance between x_{:d} and x_{:d} are too small !]".format(i, j))
                        
            lagrange_sum += grid_y[i] * basis_func

        return lagrange_sum


index = np.linspace(0, 20, 21)
x = 5.0 * np.cos((2.0*index + 1.0) / 42.0 * np.pi)
xx = np.linspace(-5, 5, 101)
y = lagrange_interp(grids = [x, R(x)], x = xx)
plt.style.use("seaborn-dark") 
fig = plt.figure()
plt.plot(x, R(x), color = "r", linewidth = 0.0, marker = "x", label = "grids")
plt.plot(xx, y, color = "b", linewidth = 1.0, linestyle = 'dashed', label = r"$L_n(x) \; (n = 20)$")
plt.plot(xx, R(xx), color = "k", label = r"$R(x) = 1 / (1+x^2)$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid()
plt.legend()
plt.savefig("./figure/lagrange_interpolation.pdf")
plt.show()


def linear_interp(grids, x):
    grid_x, grid_y = grids
    if len(grid_x) != len(grid_x):
        print("Error: Length of grid_x and grid_y must be the same !")
        
    else:
        order = np.argsort(grid_x)
        grid_x, grid_y = grid_x[order], grid_y[order]
        
        linear_sum = 0.0
        for i in range(len(grid_x)-1):
            dist = grid_x[i+1] - grid_x[i]
            if abs(dist) > 1.0e-8:
                basis_func = (grid_y[i] * (grid_x[i+1] - x) + grid_y[i+1] * (x - grid_x[i])) / (grid_x[i+1] - grid_x[i])
                func = lambda x: 1 if (x < grid_x[i+1] and x >= grid_x[i]) else 0.0
                cond = np.array(list(map(func, x)))
                linear_sum += basis_func * cond
            else: # can be optimized
                print("[Error: The distance between x_{:d} and x_{:d} are too small !]".format(i, i+1))

        return linear_sum


x = np.linspace(-5, 5, 11)
xx = np.linspace(-5, 5, 101)
y = linear_interp(grids = [x, R(x)], x = xx)
plt.style.use("seaborn-dark") 
fig = plt.figure()
plt.plot(x, R(x), color = "r", linewidth = 0.0, marker = "x", label = "grids")
plt.plot(xx, y, color = "r", linewidth = 1.0, linestyle = 'dashed', label = r"$Linear(x) \; (n = 10)$")
plt.plot(xx, R(xx), color = "k", label = r"$R(x) = 1 / (1+x^2)$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid()
plt.legend()
plt.savefig("./figure/linear_interpolation.pdf")
plt.show()


