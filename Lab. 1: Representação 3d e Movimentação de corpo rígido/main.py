# Neste laboratório, vamos aprender como plotar pontos e objetos no Matplotlib e no Plotly. Além disso, nosso objetivo nesta aula será de usar estas ferramentas para estudar o movimento de corpo rígido e a mudança de referencial!

# Exemplo de rotação e plot com Matplotlib

import numpy as np
import matplotlib.pyplot as plt
from math import pi,cos,sin

# Plot simples

x_barco = np.array([5,9,9,7,5,5])
y_barco = np.array([2,2,6,8,6,2])

plt.plot(x_barco, y_barco, linewidth=2.0)
plt.axis((-2, 12, -2, 12))
plt.title('Vista da camera')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.grid()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.show()

# Plot com setas e ponto

x_barco = np.array([5,9,9,7,5,5])
y_barco = np.array([2,2,6,8,6,2])

plt.plot(x_barco, y_barco, linewidth=2.0)
plt.plot(0,0,marker='o',color='black')
plt.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.2, linewidth=2, color='r')
plt.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.2, linewidth=2, color='g')
plt.axis((-2, 12, -2, 12))
plt.title('Vista da camera')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.grid()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.show()

# Rotação dos pontos

x_barco = np.array([5,9,9,7,5,5])
y_barco = np.array([2,2,6,8,6,2])

angulo = 45*pi/180.0
R = np.array([[cos(angulo),-sin(angulo)],[sin(angulo),cos(angulo)]])

barco = np.vstack([x_barco,y_barco])
novo_barco = R@barco

plt.plot(x_barco, y_barco, linewidth=2.0)
plt.plot(novo_barco[0,:], novo_barco[1,:], linewidth=2.0)

plt.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.2, linewidth=2, color='r')
plt.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.2, linewidth=2, color='g')
plt.axis((-2, 12, -2, 12))
plt.title('Vista da camera')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.grid()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.show()

# Exemplo de plot com Plotly

import numpy as np
import plotly.graph_objects as go
from math import pi,cos,sin

P = np.array([3,10,10])
x_objeto = 3*np.ones(17)
y_objeto = np.array([2,4,6,4,6,8,15,18,18,9.5,12,9.5,7.5,5,5,2,2])
z_objeto = np.array([5,5,7,10,12,9,17,17,14,7,4.5,3,5.5,4,2,2,5])

trace = go.Scatter3d(x=x_objeto,
                     y=y_objeto,
                     z=z_objeto,
                     mode='lines',
                     line=dict(color='blue',width=5),
                     name='Espada')

ponto = go.Scatter3d(x=np.array(P[0]),
                     y=np.array(P[1]),
                     z=np.array(P[2]),
                     mode='markers',
                     marker=dict(color='red',size=8,symbol='circle'),
                     name='P')

fig = go.Figure(data=[trace,ponto])

fig.update_layout(scene = dict(
                  xaxis_title='X',
                  yaxis_title='Y',
                  zaxis_title='Z'),
                  width=700,
                  margin=dict(r=20, b=10, l=10, t=10))

fig.show()

# Exemplo 4
# Rotação -> Translação = Pontos novos
x_rect = np.array([0, 3, 3, 0, 0])
y_rect = np.array([0, 0, 2, 2, 0])
O = np.array([9, 2])
P_rect = np.vstack([x_rect, y_rect, np.ones((1, 5))])

new_O = np.reshape(O, (-1, 1))

T_rect = np.hstack([np.identity(2), new_O])
T_rect = np.vstack([T_rect, [0, 0, 1]])

angulo = 90 * pi / 180.0
R = np.array([[cos(angulo), -sin(angulo), 0], [sin(angulo), cos(angulo), 0], [0, 0, 1]])

new_rect = T_rect @ R @ P_rect

plt.plot(x_rect, y_rect, linewidth = 2.0)
plt.plot(new_rect[0,:], new_rect[1,:], linewidth = 2.0)
plt.axis((-2, 10, -2, 10))
plt.title('Exemplo 4')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.grid()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.show()

# Exercício 2
# Translação para o ponto de referência -> Rotação -> Translação de volta para onde estava = Espada rotacionada pelo ponto P
P = np.array([3, 10, 10])

# Espada
x_objeto = 3*np.ones(17)
y_objeto = np.array([2, 4, 6, 4, 6, 8, 15, 18, 18, 9.5, 12, 9.5, 7.5, 5, 5, 2, 2])
z_objeto = np.array([5, 5, 7, 10, 12, 9, 17, 17, 14, 7, 4.5, 3, 5.5, 4, 2, 2, 5])

P_objeto = np.vstack([x_objeto, y_objeto, z_objeto, np.ones(17)])

new_O = np.reshape(P, (-1, 1)) # Linha para coluna
T_objeto = np.hstack([np.identity(3), new_O])
T_objeto = np.vstack([T_objeto, [0, 0, 0, 1]])

# Depois precisa retornar pra onde estava
old_O = np.reshape(-P, (-1, 1)) # Linha para coluna
T_minus_objeto = np.hstack([np.identity(3), old_O])
T_minus_objeto = np.vstack([T_minus_objeto, [0, 0, 0, 1]])

angulo = 5 * pi / 180.0
R = np.array([[1, 0, 0, 0], [0, cos(angulo), -sin(angulo), 0], [0, sin(angulo), cos(angulo), 0], [0, 0, 0, 1]]) # Ao redor do eixo x

new_Objeto = T_objeto @ R @ T_minus_objeto @ P_objeto

trace = go.Scatter3d(x = x_objeto,
                     y = y_objeto,
                     z = z_objeto,
                     mode = 'lines',
                     line = dict(color = 'blue', width = 5),
                     name = 'Espada')

ponto = go.Scatter3d(x = np.array(P[0]),
                     y = np.array(P[1]),
                     z = np.array(P[2]),
                     mode = 'markers',
                     marker = dict(color = 'red', size = 8, symbol = 'circle'),
                     name = 'P')
new_trace = go.Scatter3d(x = new_Objeto[0, :],
                     y = new_Objeto[1, :],
                     z = new_Objeto[2, :],
                     mode = 'lines',
                     line = dict(color = 'orange', width = 5),
                     name = 'Espada rotacionada')
fig = go.Figure(data = [trace, ponto, new_trace])

fig.update_layout(scene = dict(
                  xaxis_title = 'X',
                  yaxis_title = 'Y',
                  zaxis_title = 'Z'),
                  width = 700,
                  margin = dict(r = 20, b = 10, l = 10, t = 10))

fig.show()

# Exercício 3
# Translação para P1 e rodar 45 graus em torno do ponto P2
# Translação P1 -> Rotação -> Translação P1 = Forma rotacionada

P1 = np.array([2.0, 3.5])
P2 = np.array([5.0, 1.0])
PA = np.array([6.5, 7.0])

FX = np.array([6.5, 8.5, 6.5, 6.0, 5.0, 6.5])
FY = np.array([7.0, 5.0, 5.0, 3.5, 4.5, 7.0])

P_F = np.vstack([FX, FY, np.ones(6)])

new_O_P1 = np.reshape(P1 - PA, (-1, 1))
T_P1 = np.hstack([np.identity(2), new_O_P1])
T_P1 = np.vstack([T_P1, [0, 0, 1]])

new_O_P2 = np.reshape(P2, (-1, 1))
T_P2 = np.hstack([np.identity(2), new_O_P2])
T_P2 = np.vstack([T_P2, [0, 0, 1]])

old_O_P2 = np.reshape(-P2, (-1, 1))
T_minus_P2 = np.hstack([np.identity(2), old_O_P2])
T_minus_P2 = np.vstack([T_minus_P2, [0, 0, 1]])

old_O_P1 = np.reshape(-P1, (-1, 1))
T_minus_P1 = np.hstack([np.identity(2), old_O_P1])
T_minus_P1 = np.vstack([T_minus_P1, [0, 0, 1]])

angulo = 45 * pi / 180.0
R = np.array([[cos(angulo), -sin(angulo), 0], [sin(angulo), cos(angulo), 0], [0, 0, 1]])

new_F = T_P2 @ R @ T_minus_P2 @ T_P1 @ P_F

plt.plot(FX, FY, linewidth = 2.0)
plt.plot(new_F[0, :], new_F[1, :], linewidth = 2.0)
plt.axis((-1, 10, -1, 10))
plt.title('Exercício 3')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.grid()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.show()