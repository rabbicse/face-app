import cv2
import numpy as np
from screeninfo import get_monitors


def resize_image_to_monitor(frame):
    # Get the primary screen dimensions
    monitor = get_monitors()[0]  # Assuming you want the primary monitor
    screen_width = monitor.width
    screen_height = monitor.height
    # Set the window to fullscreen
    # Resize the image to fit the screen while maintaining aspect ratio
    img_height, img_width = frame.shape[:2]
    scale_width = float(screen_width) / float(img_width)
    scale_height = float(screen_height) / float(img_height)
    scale = min(scale_width, scale_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a black background for padding
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Center the image on the canvas
    y_offset = (screen_height - new_height) // 2
    x_offset = (screen_width - new_width) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return canvas


def draw_img(canvas, detections, window_name='Window', is_fullscreen=False):
    height, width, channels = canvas.shape
    rect_color = (224, 128, 20)
    rect_line_thickness = 5
    # draw bboxes
    for detection in detections:
        bbox = detection['bbox']
        x_min = bbox['x_min'] * width
        y_min = bbox['y_min'] * height
        x_max = bbox['x_max'] * width
        y_max = bbox['y_max'] * height
        score = bbox['score']

        # Draw the text on the image
        cv2.putText(canvas, f'Score: {score:.2f}', (int(x_min) + 10, int(y_min) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(canvas, (int(x_min), int(y_min)), (int(x_max), int(y_max)), rect_color, 2)

        cv2.line(canvas, (int(x_min), int(y_min)), (int(x_min + 15), int(y_min)), rect_color, rect_line_thickness)
        cv2.line(canvas, (int(x_min), int(y_min)), (int(x_min), int(y_min + 15)), rect_color, rect_line_thickness)

        cv2.line(canvas, (int(x_max), int(y_max)), (int(x_max - 15), int(y_max)), rect_color, rect_line_thickness)
        cv2.line(canvas, (int(x_max), int(y_max)), (int(x_max), int(y_max - 15)), rect_color, rect_line_thickness)

        cv2.line(canvas, (int(x_max - 15), int(y_min)), (int(x_max), int(y_min)), rect_color, rect_line_thickness)
        cv2.line(canvas, (int(x_max), int(y_min)), (int(x_max), int(y_min + 15)), rect_color, rect_line_thickness)

        cv2.line(canvas, (int(x_min), int(y_max - 15)), (int(x_min), int(y_max)), rect_color, rect_line_thickness)
        cv2.line(canvas, (int(x_min), int(y_max)), (int(x_min + 15), int(y_max)), rect_color, rect_line_thickness)

        # Draw landmarks
        landmarks = detection['landmarks']
        # for landmark in landmarks:
        eye_color = (0, 0, 255)
        nose_color = (0, 255, 0)
        lip_color = (255, 0, 0)
        draw_landmark(canvas, landmarks['left_eye'], eye_color, width, height)
        draw_landmark(canvas, landmarks['right_eye'], eye_color, width, height)
        draw_landmark(canvas, landmarks['nose'], nose_color, width, height)
        draw_landmark(canvas, landmarks['left_lip'], lip_color, width, height)
        draw_landmark(canvas, landmarks['right_lip'], lip_color, width, height)

    if is_fullscreen:
        cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, canvas)


def draw_landmark(canvas, landmark, circle_color, width, height, radius:int = 3):
    x = landmark['x'] * width
    y = landmark['y'] * height
    cv2.circle(canvas, (int(x), int(y)), radius, circle_color, -1, lineType=cv2.LINE_AA)  # Blue dots
