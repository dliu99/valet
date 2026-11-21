"""Finding the table tag in a wrist frame and turning it into the camera's pose in the world."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from valet_sim.scene import TagSpec, tag_corners

from .camera import Lens, pose

_DETECTOR = None


def _detector() -> cv2.aruco.ArucoDetector:
    global _DETECTOR
    if _DETECTOR is None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
    return _DETECTOR


@dataclass(frozen=True)
class Sighting:
    corners_px: np.ndarray  # 4x2, OpenCV's order
    T_world_cam: np.ndarray
    reprojection_px: float
    side_px: float  # mean edge length in the frame: how big the tag looks


def find(frame: np.ndarray, tag: TagSpec) -> np.ndarray | None:
    gray = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if int(i) == tag.id:
            return c[0].astype(float)
    return None


def camera_from_tag(corners_px: np.ndarray, tag: TagSpec, lens: Lens = Lens()) -> tuple[np.ndarray, float]:
    """Camera-to-world from the tag's four corners (IPPE on the undistorted corners), and its reprojection error."""
    world = tag_corners(tag)
    centre = world.mean(0)
    obj = (world - centre).astype(np.float64)
    obj[:, 2] = 0.0
    img = lens.undistort(corners_px).astype(np.float64)
    best = None
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    for r, t in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(r)
        T_cam_tag = pose(R, t.ravel())
        T_world_cam = pose(np.eye(3), centre) @ np.linalg.inv(T_cam_tag)
        err = float(np.abs(lens.project(T_world_cam, world) - corners_px).max())
        if T_world_cam[2, 3] > 0 and (best is None or err < best[1]):
            best = (T_world_cam, err)
    if best is None:
        raise ValueError("no pose puts the camera above the table")
    return best


def sight(frame: np.ndarray, tag: TagSpec, lens: Lens = Lens()) -> Sighting | None:
    corners = find(frame, tag)
    if corners is None:
        return None
    T, err = camera_from_tag(corners, tag, lens)
    side = float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)))
    return Sighting(corners, T, err, side)
