"""
Head pose convention shared by every client (matches Google ML Kit, so the Flutter app can use
``Face.headEulerAngle{X,Y,Z}`` directly). All angles are in degrees, measured on the
*non-mirrored* camera frame:

* ``yaw``   > 0: face turned towards the image right (= the user's own left)
* ``pitch`` > 0: face looking up
* ``roll``  > 0: face rotated counter-clockwise in the image
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float


def pose_from_landmarks_3d(left_eye: Sequence[float], right_eye: Sequence[float],
                           forehead: Sequence[float], chin: Sequence[float]) -> HeadPose:
    """
    Pose from 4 points in pixel units (x right, y down, z away from the camera), e.g. MediaPipe
    face mesh points 33 (image-left eye), 263 (image-right eye), 10 (forehead) and 152 (chin).
    """
    le, re, fh, ch = (np.asarray(p, dtype=np.float64) for p in (left_eye, right_eye, forehead, chin))
    x_axis = re - le
    down = ch - fh
    x_axis /= np.linalg.norm(x_axis)
    # remove the x component so the axes are orthogonal
    down = down - x_axis * np.dot(down, x_axis)
    down /= np.linalg.norm(down)
    forward = np.cross(down, x_axis)  # towards the camera for a frontal face: (0, 0, -1)

    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -forward[1]))))
    roll = math.degrees(math.atan2(-x_axis[1], x_axis[0]))
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


# Registration pose targets: label -> (yaw range, pitch range, max |roll|).
# Clients evaluate the non frontal targets relative to the user's mean frontal pose, which cancels
# the camera placement (laptop camera below the eyes, phone held low, ...).
POSE_TARGETS = {
    "frontal": ((-12.0, 12.0), (-15.0, 15.0), 10.0),
    "turn_left": ((15.0, 40.0), (-15.0, 15.0), 15.0),
    "turn_right": ((-40.0, -15.0), (-15.0, 15.0), 15.0),
    "look_up": ((-12.0, 12.0), (10.0, 30.0), 15.0),
    "look_down": ((-12.0, 12.0), (-30.0, -10.0), 15.0),
}


def classify_pose(pose: HeadPose) -> Optional[str]:
    for label, ((y0, y1), (p0, p1), max_roll) in POSE_TARGETS.items():
        if y0 <= pose.yaw <= y1 and p0 <= pose.pitch <= p1 and abs(pose.roll) <= max_roll:
            return label
    return None
