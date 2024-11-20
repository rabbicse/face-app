import cv2


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


def draw_landmark(canvas, landmark, circle_color, width, height):
    x = landmark['x'] * width
    y = landmark['y'] * height
    cv2.circle(canvas, (int(x), int(y)), 5, circle_color, -1, lineType=cv2.LINE_AA)  # Blue dots
