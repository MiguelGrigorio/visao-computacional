# Neste laboratório, vamos testar nossos conhecimentos sobre transformações de homografia! Para isso, vamos desenvolver nossa própria função de homografia usando o algoritmo DLT. É possível utilizar uma função já pronta do OpenCV, entretanto, nosso objetivo é criar a nossa e depois comparar com a dele!

import numpy as np
import cv2 as cv                             # OpenCV
import plotly.graph_objects as go            # Plotly
from plotly.subplots import make_subplots    # Subplots no Plotly
import plotly.express as px                  # Plotar imagens com o Plotly
from matplotlib import pyplot as plt

# Definição do DLT

# H_matrix transforma de pts1 para pts2
def meu_DLT(pts1,pts2):

  # Adicionar a coordenada homogenea nos pontos
  pts1h = np.vstack([pts1, np.ones(pts1.shape[1])])
  pts2h = np.vstack([pts2, np.ones(pts2.shape[1])])

  # Calcula a matriz A
  A = np.zeros((2 * pts1.shape[1], 9)) # 8x9 (nesse caso)
  k = 0
  for i in range(0, pts1.shape[1]): # faz para cada ponto um Ai só que junta
    A[k,3:6] = -pts2h[2,i] * pts1h[:,i].T # -w2i * pt1[i]
    A[k,6:9] = pts2h[1,i] * pts1h[:,i].T  # y2i * pt1[i]
    A[k+1,0:3] = pts2h[2,i] * pts1h[:,i].T # w2i * pt1[i]
    A[k+1,6:9] = -pts2h[0,i] * pts1h[:,i].T # w2i * pt1[i]
    k = k+2
  # Calcula o SVD(A) = U.S.Vt
  U,S,Vt = np.linalg.svd(A)

  # Reshape da ultima linha de Vt (ultima coluna de V) para a matriz de homografia
  h = Vt[-1,:]
  H_matrix = h.reshape(3,3)

  return H_matrix

# Mudando o ponto de vista por homografia (Plotly)

# Le a imagem e define os cantos
img1 = cv.imread('comicsStarWars02.jpg',0)

corners_img1 = np.array([[105,123],[650,55],[580,1055],[58,920]])
corners_img2 = np.array([[58,123],[650,123],[650,1000],[58,1000]])

src_pts = np.float32(corners_img1)
dst_pts = np.float32(corners_img2)

# Subtitua a funcao do OpenCV pela sua funcao de homografia
#H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
H = meu_DLT(src_pts.T, dst_pts.T)

# Warp da imagem e plot
img2 = cv.warpPerspective(img1, H, (img1.shape[1],img1.shape[0]))

fig = make_subplots(rows=1, cols=2)
fig.add_trace(px.imshow(img1, color_continuous_scale='gray').data[0], row=1, col=1)
fig.add_trace(px.imshow(img2, color_continuous_scale='gray').data[0], row=1, col=2)
fig.add_trace(go.Scatter(x=corners_img2[:,0],y=corners_img2[:,1],mode='markers',marker=dict(color='blue',size=7,symbol='star')), row=1, col=1)
fig.add_trace(go.Scatter(x=corners_img1[:,0],y=corners_img1[:,1],mode='markers',marker=dict(color='red',size=7,symbol='star')), row=1, col=1)
fig.add_trace(go.Scatter(x=corners_img2[:,0],y=corners_img2[:,1],mode='markers',marker=dict(color='blue',size=7,symbol='star')), row=1, col=2)
fig.update_yaxes(autorange='reversed')
fig.update_layout(
    autosize=False,
    width=1000,  # Largura total da figura
    height=700,  # Altura total da figura
    xaxis=dict(
        scaleanchor='y',  # Ancorar a escala do eixo x à escala do eixo y
        scaleratio=1,     # Manter a proporção
    ),
    xaxis2=dict(
        scaleanchor='y2',  # Ancorar a escala do eixo x2 à escala do eixo y2
        scaleratio=1,     # Manter a proporção
    ),
    coloraxis_showscale=False  # Ocultar a barra de cores
)
fig.show()

# Mudando o ponto de vista por homografia (Matplotlib)

# Le a imagem e define os cantos
img1 = cv.imread('comicsStarWars02.jpg',0)

corners_img1 = np.array([[105,123],[650,55],[580,1055],[58,920]])
corners_img2 = np.array([[58,123],[650,123],[650,1000],[58,1000]])

src_pts = np.float32(corners_img1)
dst_pts = np.float32(corners_img2)

# Subtitua a funcao do OpenCV pela sua funcao de homografia
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
#H = meu_DLT(src_pts.T, dst_pts.T)

# Warp da imagem e plot
img2 = cv.warpPerspective(img1, H, (img1.shape[1],img1.shape[0]))

fig = plt.figure()
fig, axs = plt.subplots(1,2,figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)
plt.imshow(img1, 'gray')
plt.plot(corners_img2[:,0],corners_img2[:,1],'*b')
plt.plot(corners_img1[:,0],corners_img1[:,1],'*r')
ax2 = fig.add_subplot(1,2,2)
plt.imshow(img2, 'gray')
plt.plot(corners_img2[:,0],corners_img2[:,1],'*b')
plt.show()

# Algoritmo DLT normalizado

# Vamos agora alterar nosso algoritmo de homografia para que o DLT tenha seus pontos normalizados na estimativa de H.

# Definição do DLT normalizado

# H_matrix transforma de pts1 para pts2
def meu_DLT(pts1,pts2):

  # Adicionar a coordenada homogenea nos pontos
  pts1h = np.vstack([pts1,np.ones(pts1.shape[1])])
  pts2h = np.vstack([pts2,np.ones(pts2.shape[1])])

  # Calcula a matriz A
  A = np.zeros((2*pts1.shape[1],9))
  k = 0
  for i in range(0,pts1.shape[1]):
    A[k,3:6] = -pts2h[2,i]*pts1h[:,i].T #-w2i*pt1[i]
    A[k,6:9] = pts2h[1,i]*pts1h[:,i].T  #y2i*pt1[i]
    A[k+1,0:3] = pts2h[2,i]*pts1h[:,i].T #w2i*pt1[i]
    A[k+1,6:9] = -pts2h[0,i]*pts1h[:,i].T #w2i*pt1[i]
    k = k+2

  # Calcula o SVD(A) = U.S.Vt
  U,S,Vt = np.linalg.svd(A)

  # Reshape da ultima linha de Vt (ultima coluna de V) para a matriz de homografia
  h = Vt[-1,:]
  H_matrix = h.reshape(3,3)

  return H_matrix

def normaliza_pontos(pontos):

  # Calcular o centroide
  centro = np.mean(pontos,axis=1).reshape(-1, 1)

  # Calcular a distancia media dos pontos tendo o centroide como origem
  novos_pontos = pontos - centro
  dist_pontos = np.sqrt(novos_pontos[:,0]**2+novos_pontos[:,1]**2)
  dist_media = np.mean(dist_pontos)

  # Definir a escala para ter como distancia media sqrt(2)
  esc = np.sqrt(2)/dist_media

  # Definir a matriz de normalizacao
  T = np.array([[esc, 0 , -esc*centro[0][0]],[0, esc, -esc*centro[1][0]], [0, 0, 1]])

  # Coordenadas homogeneas
  pontosh = np.vstack((pontos,np.ones(pontos.shape[1])))
  # Normalizacao dos pontos
  npontosh = T@pontosh

  # Reshape dos pontos para o formato original, eliminando a coordenada homogenea
  npontosh = npontosh[:-1,:]

  return T, npontosh

def minha_homografia(pts1,pts2):

  # Normaliza pontos
  T1, npts1 = normaliza_pontos(pts1)
  T2, npts2 = normaliza_pontos(pts2)

  # Execucao do DLT e obtencao da H normalizada
  Hn = meu_DLT(npts1,npts2)

  # Desnormalizacao da matriz de homografia H
  H = np.linalg.inv(T2) @ Hn @ T1

  return H

# Teste agora o seu algoritmo do DLT normalizado no warp das imagens anteriores!

# Le a imagem e define os cantos
img1 = cv.imread('comicsStarWars02.jpg',0)

corners_img1 = np.array([[105,123],[650,55],[580,1055],[58,920]])
corners_img2 = np.array([[58,123],[650,123],[650,1000],[58,1000]])

src_pts = np.float32(corners_img1)
dst_pts = np.float32(corners_img2)

# Subtitua a funcao do OpenCV pela sua funcao de homografia
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
H = minha_homografia(src_pts.T, dst_pts.T)

# Warp da imagem e plot
img2 = cv.warpPerspective(img1, H, (img1.shape[1],img1.shape[0]))

fig = make_subplots(rows=1, cols=2)
fig.add_trace(px.imshow(img1, color_continuous_scale='gray').data[0], row=1, col=1)
fig.add_trace(px.imshow(img2, color_continuous_scale='gray').data[0], row=1, col=2)
fig.add_trace(go.Scatter(x=corners_img2[:,0],y=corners_img2[:,1],mode='markers',marker=dict(color='blue',size=7,symbol='star')), row=1, col=1)
fig.add_trace(go.Scatter(x=corners_img1[:,0],y=corners_img1[:,1],mode='markers',marker=dict(color='red',size=7,symbol='star')), row=1, col=1)
fig.add_trace(go.Scatter(x=corners_img2[:,0],y=corners_img2[:,1],mode='markers',marker=dict(color='blue',size=7,symbol='star')), row=1, col=2)
fig.update_yaxes(autorange='reversed')
fig.update_layout(
    autosize=False,
    width=1000,  # Largura total da figura
    height=700,  # Altura total da figura
    xaxis=dict(
        scaleanchor='y',  # Ancorar a escala do eixo x à escala do eixo y
        scaleratio=1,     # Manter a proporção
    ),
    xaxis2=dict(
        scaleanchor='y2',  # Ancorar a escala do eixo x2 à escala do eixo y2
        scaleratio=1,     # Manter a proporção
    ),
    coloraxis_showscale=False  # Ocultar a barra de cores
)
fig.show()