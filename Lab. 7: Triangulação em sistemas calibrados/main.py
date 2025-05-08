import cv2
import time
import numpy as np
from utils import camera_parameters, get_image

calibrations_path = "calibrations/*.npz"

cams_params = camera_parameters(calibrations_path)

# Caso queira anotar o pixel de cada imagem
show = False
if show:
    n_cams = 4
    for camera_id in range(1, n_cams+1):
        get_image(camera_id)
        time.sleep(2)

# Exemplo
pontos_camera_original = {
    1: (630, 350),
    2: (610, 280),
    3: (710, 300),
    4: (700, 260),
}

def undistort_points(parameters: dict, original_points: tuple | list) -> list:
    """
    A função retira a distorção do ponto na câmera selecionada.
    """
    K = parameters['K']
    Rtcw = parameters['rt']
    dist = parameters['distortion']
    nK = parameters['nK']

    undistorted_points = cv2.undistortPoints(original_points, K, dist, None, nK).squeeze(axis=0)

    P = nK @ Rtcw

    return P, undistorted_points

def point2world(parameters: dict, original_points: dict, selectCameras: list) -> list:
    """
    A função considera que os pontos recebidos estão distorcidos pela câmera, então passa pela função `undistort_points` primeiro.
    """
    
    A = []
    for id in selectCameras:
        P, undistorted_points = undistort_points(parameters[id], original_points[id])
        unds = undistorted_points[0]

        u = np.array([[int(unds[0]), int(unds[1]), 1]]).T
        zeros = np.zeros((3, 1))
        mtx = np.array(P)

        # Formando as linhas
        for i in selectCameras:
            if i == id:
                mtx = np.hstack((mtx, -u))
            else:
                mtx = np.hstack((mtx, zeros))

        # Formando a matriz A
        if len(A) == 0:
            A = mtx
        else:
            A = np.vstack((A, mtx))

    # Realizando SVD
    _, _, V_transpose = np.linalg.svd(A)
    V = V_transpose[-1]
    V = V / V[3]
    Xw = V[:3]
    return Xw

print(point2world(cams_params, pontos_camera_original, [1, 2, 3, 4]))