# Partes geradas por IA para substituir o interact

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import trimesh
import numpy as np
from math import pi, cos, sin

def translacao(dx):
    T = np.eye(len(dx)+1)
    for i, d in enumerate(dx):
        T[i, -1] = d
    return T

def rotacao(angulo, dim='2d'):
    anguloD = angulo * pi / 180
    if dim == '2d':
        R = np.array([[cos(anguloD), -sin(anguloD), 0],
                      [sin(anguloD), cos(anguloD), 0],
                      [0, 0, 1]])
    elif dim == 'x':
        R = np.array([[1, 0, 0, 0],
                      [0, cos(anguloD), -sin(anguloD), 0],
                      [0, sin(anguloD), cos(anguloD), 0],
                      [0, 0, 0, 1]])
    elif dim == 'y':
        R = np.array([[cos(anguloD), 0, sin(anguloD), 0],
                      [0, 1, 0, 0],
                      [-sin(anguloD), 0, cos(anguloD), 0],
                      [0, 0, 0, 1]])
    else:
        R = np.array([[cos(anguloD), -sin(anguloD), 0, 0],
                      [sin(anguloD), cos(anguloD), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    return R

class Camera:
    def __init__(self, focal=1, cam_center=np.array([0, 0]), cam_pos=np.array([0, 0, 0]), cam_rot=np.array([0, 0, 0])):
        self.K = np.array([[focal, 0, cam_center[0]],
                           [0, focal, cam_center[1]],
                           [0, 0, 1]])
        self.R = rotacao(cam_rot[2], dim='z') @ rotacao(cam_rot[1], dim='y') @ rotacao(cam_rot[0], dim='x')
        self.t = translacao(cam_pos)
        self.Rt = np.linalg.inv(self.t @ self.R)
        self.P = self.K @ np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]]) @ self.Rt

mesh = trimesh.load_mesh("frog.obj", process=False)
vertices = np.array(mesh.vertices).T
vertices = np.vstack([vertices, np.ones((1, vertices.shape[1]))])
vertices_orig = vertices

def update_plot():
    x = x_slider.get()
    y = y_slider.get()
    z = z_slider.get()
    xc = xc_slider.get()
    yc = yc_slider.get()
    zc = zc_slider.get()
    rx = rx_slider.get()
    ry = ry_slider.get()
    rz = rz_slider.get()
    f = focal_slider.get()

    cam = Camera(cam_pos=np.array([xc, yc, zc]),
                 cam_rot=np.array([rx, ry, rz]),
                 focal=f,
                 cam_center=np.array([319, 239]))

    T = translacao(np.array([x, y, z]))
    vertices = T @ vertices_orig
    pontos_graph = cam.P @ vertices

    points = [[], [], []]
    for p in range(len(pontos_graph[2])):
        if pontos_graph[2][p] > 0:
            points[0].append(pontos_graph[0][p])
            points[1].append(pontos_graph[1][p])
            points[2].append(pontos_graph[2][p])

    points = np.array(points)
    points = points / points[2, :]

    ax1.clear()
    ax2.clear()
    ax1.set_title("3D View")
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_zlim([-5, 5])
    ax1.scatter(vertices[0, :], vertices[1, :], vertices[2, :], color='black', s=1)

    ax2.set_title("2D Projection")
    ax2.set_xlim([0, 639])
    ax2.set_ylim([479, 0])
    ax2.scatter(points[0, :], points[1, :], color='black', s=1)

    canvas.draw()

root = tk.Tk()
root.title("Projeção com Tkinter")

frame_plot = tk.Frame(root)
frame_plot.grid(row=0, column=0)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas.get_tk_widget().pack()

frame_controls = tk.Frame(root)
frame_controls.grid(row=1, column=0)

def add_slider(label, from_, to, row, init):
    tk.Label(frame_controls, text=label).grid(row=row, column=0)
    s = tk.Scale(frame_controls, from_=from_, to=to, orient=tk.HORIZONTAL, resolution=1, command=lambda e: update_plot())
    s.set(init)
    s.grid(row=row, column=1)
    return s

x_slider = add_slider("x", -10, 10, 0, 0)
y_slider = add_slider("y", -10, 10, 1, 0)
z_slider = add_slider("z", -10, 10, 2, 0)
xc_slider = add_slider("xc", -15, 15, 3, 0)
yc_slider = add_slider("yc", -15, 15, 4, 0)
zc_slider = add_slider("zc", -15, 15, 5, 5)
rx_slider = add_slider("rx", -180, 180, 6, -90)
ry_slider = add_slider("ry", -180, 180, 7, 0)
rz_slider = add_slider("rz", -180, 180, 8, 0)
focal_slider = add_slider("focal", 50, 500, 9, 100)

update_plot()
root.mainloop()
