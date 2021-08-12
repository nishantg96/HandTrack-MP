import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time
import tensorflow as tf

# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cv2.CAP_PROP_FPS = 60
print(cv2.CAP_PROP_FPS)

prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Flip on horizontal
        image = cv2.flip(frame, 1)
        # Set flag
        image.flags.writeable = False
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        cv2.putText(image, fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
        # Detections
        

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        b_channel, g_channel, r_channel = cv2.split(image)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
        alpha_channel = alpha_channel.astype(np.uint8)
        image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                for point in mp_hands.HandLandmark:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(16, 0, 255), thickness=4, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(255, 0, 16), thickness=5, circle_radius=2),
                                              )
                    imageHeight, imageWidth, _ = image.shape

                    normalizedLandmark = hand.landmark[point]
                    pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                              normalizedLandmark.y,
                                                                                              imageWidth, imageHeight)

                    # print(point)
                    # print(pixelCoordinatesLandmark)
                    # print(normalizedLandmark)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()