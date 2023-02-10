import cv2 as cv
import sys


# access webcam with cv2
def access_webcam():
    webcam = cv.VideoCapture(0)

    while True:
        successful_frame_read, frame = webcam.read()
        cv.imshow('Webcam', frame)
        cv.waitKey(1)
# access other video with cv2
def access_video():
    video = cv.VideoCapture('video.mp4')

    while True:
        successful_frame_read, frame = video.read()
        cv.imshow('Video', frame)
        cv.waitKey(1)

def access_image():
    image = cv.imread('image.jpg')
    cv.imshow('Image', image)
    cv.waitKey(0)