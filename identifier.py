import mediapipe as mp
import cv2
import sys
import cam


# ID class mediapipe holistic image detection
class Identifier:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic()

    def identify(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        return image

    # draw landmarks on webcam
    def draw_landmarks_webcam(self):
        webcam = cam.access_webcam()
        while True:
            successful_frame_read, frame = webcam.read()
            image, results = self.identify(frame)
            image = self.draw_landmarks(image, results)
            cv2.imshow('Webcam', image)
            cv2.waitKey(1)

    # draw landmarks on video
    def draw_landmarks_video(self):
        video = cam.access_video()
        while True:
            successful_frame_read, frame = video.read()
            image, results = self.identify(frame)
            image = self.draw_landmarks(image, results)
            cv2.imshow('Video', image)
            cv2.waitKey(1)

    # draw landmarks on image
    def draw_landmarks_image(self, image):
        image, results = self.identify(image)
        image = self.draw_landmarks(image, results)
        return image



#display webcam with landmarks
if sys.argv[1] == 'webcam':
    Identifier().draw_landmarks_webcam()

