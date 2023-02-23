import mediapipe as mp
import cv2
import sys
import os
import stdlib_list


class profile:

    def __init__(self, name, landmarks):
        self.name = name
        self.landmarks = landmarks

