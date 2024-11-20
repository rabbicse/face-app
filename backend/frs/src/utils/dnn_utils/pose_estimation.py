import math
import cv2.cv2 as cv2
import numpy as np

"""
https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV
"""

def to_numpy(landmarks):
    """
    @param landmarks:
    @return:
    """
    coords = []
    for l in landmarks:
        coords += [[l[0], l[1]]]
    return np.array(coords).astype(np.double)


class PoseEstimation:
    model_points = np.array([
        (2.37427, 110.322, 21.7776),  # Left eye left corner
        (70.0602, 109.898, 20.8234),  # Right eye right corne
        (36.8301, 78.3185, 52.0345),  # Nose tip
        (14.8498, 51.0115, 30.2378),  # Left Mouth corner
        (58.1825, 51.0115, 29.6224)  # Right mouth corner
    ])

    def __init__(self):
        pass

    def estimate(self, pic, bbox, lm, is_draw=False):
        try:
            landmarks = [lm[0],
                         lm[1],
                         lm[2],
                         lm[3],
                         lm[4]]
            imgpts, modelpts, roll, pitch, yaw, nose = self.face_orientation(pic, to_numpy(landmarks))

            return abs(roll) < 25 and abs(pitch) < 25 and abs(yaw) < 20

            # todo: visualize
            # if is_draw:
            #     i = 0
            #     for p in landmarks:
            #         cv2.circle(pic, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
            #
            #     poses = ['Roll', 'Pitch', 'Yaw']
            #     for j in range(len(rotate_degree)):
            #         text = '{}: {:05.2f}'.format(poses[j], float(rotate_degree[j]))
            #         cv2.putText(pic, text, (int(bbox[0]), int(bbox[1] + (20 * j))), cv2.FONT_HERSHEY_SIMPLEX,
            #                     0.5, (255, 255, 255), thickness=1, lineType=1)
            #
            #     print(pic.shape)
            #     thickness = 2
            #     print(nose)
            #     print(imgpts[1].ravel())
            #     n = (int(lm[2][0]), int(lm[2][1]))
            #     print(n)
            #     i0 = tuple(imgpts[1].ravel())
            #     cv2.line(pic, n, (int(i0[0]), int(i0[1])), (0, 255, 0), thickness)  # GREEN
            #     # cv2.line(pic, tuple(nose), tuple(imgpts[1].ravel()), (0, 255, 0), thickness)  # GREEN
            #     # cv2.line(pic, tuple(nose), tuple(imgpts[0].ravel()), (255, 0, 0), thickness)  # BLUE
            #     # cv2.line(pic, tuple(nose), tuple(imgpts[2].ravel()), (0, 0, 255), thickness)  # RED
            #
            #     cv2.imshow('img', pic)
            #     cv2.waitKey(0)
            #
            #     # self.draw_pose(img, bbox, to_numpy(landmarks))
            #
            #     # draw arrows
            #     self.draw_arrow(pic, tuple(nose), tuple(imgpts[0].ravel()), (255, 0, 0))
            #     self.draw_arrow(pic, tuple(nose), tuple(imgpts[1].ravel()), (0, 255, 0))
            #     self.draw_arrow(pic, tuple(nose), tuple(imgpts[2].ravel()), (0, 0, 255))
            #
            #     cv2.imshow('img', pic)
            #     cv2.waitKey()
        except Exception as x:
            print(x)

    @staticmethod
    def draw_arrow(img, p, q, color):
        """
        @param img:
        @param p:
        @param q:
        @param color:
        @return:
        """
        arrow_magnitude = 9
        angle = math.atan2(p[1] - q[1], p[0] - q[0])
        px = q[0] + arrow_magnitude * math.cos(angle + math.pi / 4)
        py = q[1] + arrow_magnitude * math.sin(angle + math.pi / 4)
        cv2.line(img, (int(px), int(py)), q, color, 3)

        px = q[0] + arrow_magnitude * math.cos(angle - math.pi / 4)
        py = q[1] + arrow_magnitude * math.sin(angle - math.pi / 4)
        cv2.line(img, (int(px), int(py)), q, color, 3)

    def face_orientation(self, frame, landmarks):
        """
        @param frame:
        @param landmarks:
        @return:
        """
        image_points = np.array([
            (landmarks[0]),  # Left eye left corner
            (landmarks[1]),  # Right eye right corner
            (landmarks[2]),  # Nose tip
            (landmarks[3]),  # Left Mouth corner
            (landmarks[4])  # Right mouth corner
        ], dtype="double")

        # get frame width height and channel number
        row, column, channel = frame.shape  # (height, width, color_channel)

        # max distance
        max_d = (row + column) / 2

        # set camera matrix
        camera_matrix = np.array(
            [[max_d, 0, column / 2.0],
             [0, max_d, row / 2.0],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points,
                                                                      image_points,
                                                                      camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

        # set axis
        l = 80
        x = self.model_points[2][0]
        y = self.model_points[2][1]
        z = self.model_points[2][2]
        axis = np.float32([[x + l, y, z],
                           [z, y + l, z],
                           [z, y, z + l]])

        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        modelpts, jac2 = cv2.projectPoints(self.model_points, rotation_vector, translation_vector, camera_matrix,
                                           dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        print(f'Roll: {roll} - Pitch: {pitch} - Yaw: {yaw}')

        return imgpts, modelpts, roll, pitch, yaw, landmarks[2]