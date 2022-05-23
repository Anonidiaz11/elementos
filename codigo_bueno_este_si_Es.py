# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:20:24 2022

@author: Julian Diaz
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import meshio
import math
plt.rcParams["mathtext.fontset"] = "cm"
def assem(coords, elems, source):
    """
    Ensambla la matriz de rigidez y el vector de cargas
    
    Parámetros
    ----------
    coords : ndarray, float
        Coordenadas de los nodos.
    elems : ndarray, int
        Conectividad de los elementos.
    source : ndarray, float
        Término fuente evaluado en los nodos de la malla.
    
    Retorna
    -------
    stiff : ndarray, float
        Matriz de rigidez del problema.
    mass : ndarray, float
        Matriz de masa del problema.
    rhs : ndarray, float
        Vector de cargas del problema.
    """
    ncoords = coords.shape[0]
    stiff = np.zeros((ncoords, ncoords))
    mass = np.zeros((ncoords, ncoords))
    rhs = np.zeros((ncoords))
    for el_cont, elem in enumerate(elems):
        stiff_loc, mass_loc, det = local_mat(coords[elem])
        rhs[elem] += det*np.mean(source[elem])/6
        for row in range(3):
            for col in range(3):
                row_glob, col_glob = elem[row], elem[col]
                stiff[row_glob, col_glob] += stiff_loc[row, col]
                mass[row_glob, col_glob] += mass_loc[row, col]
    return stiff, mass, rhs
def local_mat(coords):
    """Calcula la matriz de rigidez local
        
    Parámetros
    ----------
    coords : ndarray, float
        Coordenadas de los nodos del elemento.
    
    Retorna
    -------
    stiff : ndarray, float
        Matriz de rigidez local.
    det : float
        Determinante del jacobian.
    mass : float
        Matriz de masa local.
    """
    dNdr = np.array([
            [-1, 1, 0],
            [-1, 0, 1]])
    jaco = dNdr @ coords
    det = np.linalg.det(jaco)
    jaco_inv = np.linalg.inv(jaco)
    dNdx = jaco_inv @ dNdr
    stiff = 0.5 * det * (dNdx.T @ dNdx)
    mass = det/24 * np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]])
    return stiff, mass, det
def pts_restringidos(coords, malla, lineas_rest):
    """
    Identifica los nodos restringidos y libres
    para la malla.
        
    Parámetros
    ----------
    coords : ndarray, float
        Coordenadas de los nodos del elemento.
    malla : Meshio mesh
        Objeto de malla de Meshio.
    lineas_rest : list
        Lista con los números para las líneas
        restringidas.
    
    Retorna
    -------
    pts_rest : list
        Lista de puntos restringidos.
    pts_libres : list
        Lista de puntos libres.
    """
    lineas = [malla.cells[k].data for k in lineas_rest]
    pts_rest = []
    for linea in lineas:
        pts_rest += set(linea.flatten())
    pts_libres = list(set(range(coords.shape[0])) - set(pts_rest))
    return pts_rest, pts_libres
malla = meshio.read("conectividad.msh")
pts = malla.points
fila_ele, colu_ele = pts.shape
fuente=np.zeros((fila_ele,1))
x, y = pts[:, 0:2].T
tri = malla.cells[0].data
plt.figure()
plt.triplot(x, y, tri, linewidth=0.2)
plt.axis("image")
plt.show()
for j in range(0,fila_ele):
    fuente[j,0]=np.sin(pts[j,0])*np.sin(pts[j,1])
stiff, mass, rhs = assem(pts[:, :2], tri, fuente)   
u_ini = np.sin(x)*np.sin(y-2)
fig0 = plt.figure()
ax0 = fig0.add_subplot(projection='3d')
ax0.plot_trisurf(x, y, u_ini, triangles=tri, cmap="viridis")
ax0.set_zlim(0, 1000)

ax0.set_xlabel(r"$x$", fontsize=16)
ax0.set_ylabel(r"$y$", fontsize=16)
ax0.set_zlabel(r"$u_0$", fontsize=16)
pts_rest, pts_libres = pts_restringidos(pts, malla, [0])

plt.show()
dt = 0.001
niter = 50
u_total = np.zeros((x.shape[0], niter))
u_total[:, 0] = u_ini

A = mass[np.ix_(pts_libres, pts_libres)] + dt*stiff[np.ix_(pts_libres, pts_libres)]

for cont in range(1, niter):
    b = (mass @ u_total[:, cont-1])[pts_libres] - dt * rhs[pts_libres]
    sol_aux = np.linalg.solve(A, b)
    u_total[pts_libres, cont] = sol_aux
    import os
carpeta = "./cuadrado_anim/"
if os.path.isdir(carpeta):
    pass
else:
    os.mkdir(carpeta)
for cont in range(niter):
    malla.point_data["solucion"] = u_total[:, cont]
    malla.write(carpeta + "cuadrado_anim" + str(cont).zfill(2) + ".vtk")