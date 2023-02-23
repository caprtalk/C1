import face_identification
import sys
import cam
import profile


# main function
def main():
    # display webcam with landmarks
    if sys.argv[1] == 'webcam':
        identifier.Identifier().draw_landmarks_webcam()

if __name__ == '__main__':
    main()