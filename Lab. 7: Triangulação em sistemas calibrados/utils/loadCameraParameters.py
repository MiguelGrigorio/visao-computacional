import glob
import numpy as np
from typing import Dict

def load_camera_parameters(calibration_path: str, camera_id: int) -> Dict[str, np.ndarray | int | tuple | float]:
    """
    Exemplo de retorno:
    >>> {
            "camera_id": 1,
            "K": [...],
            "resolution": (1280, 720),
            "rt": [...],
            "nK": [...],
            "distortion": [...],
            "roi": [...],
            "scale": 1.0
    }
    """

    camera_data = np.load(calibration_path)
    cam_calib: Dict[str, np.ndarray | int | tuple | float] = {
        "camera_id": camera_id,
        "K": camera_data['K'],
        "resolution": (int(camera_data["w"]), int(camera_data["h"])),
        "rt": camera_data["rt"],
        "nK": camera_data["nK"],
        "distortion": camera_data["dist"],
        "roi": camera_data["roi"],
        "scale": float(camera_data["escala"])
    }
    return cam_calib

def camera_parameters(calibrations_path: str = "calibrations/*.npz") -> Dict[int, Dict[str, np.ndarray | int | tuple | float]]:
    """
    Caso os arquivos em `calibrations_path` já estiverem em ordem e completos só rodar essa função.\n
    Exemplo de retorno:\n
    >>> camera_parameters()
    {
        1: {
            "camera_id": 1,
            "K": [...],
            "resolution": (1280, 720),
            "rt": [...],
            "nK": [...],
            "distortion": [...],
            "roi": [...],
            "scale": 1.0
        },
        2: {
            "camera_id": 2,
            "K": [...],
            "resolution": (1280, 720),
            "rt": [...],
            "nK": [...],
            "distortion": [...],
            "roi": [...],
            "scale": 1.5
        }
    }
    """
    cams_calib: Dict[int, Dict[str, np.ndarray | int | tuple | float]] = {}
    for idx, path in enumerate(glob.glob(calibrations_path)):
        cams_calib[idx + 1] = load_camera_parameters(path, idx + 1)
    return cams_calib