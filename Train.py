# Stamp: 32:39
import mediapipe
import cv2
import os
import csv
import numpy
import pandas
import pickle
import warnings

warnings.filterwarnings("ignore")

mpDrawingUtils = mediapipe.solutions.drawing_utils
mpHolisitcModel = mediapipe.solutions.holistic

# Configuring the holistic model
holistic = mpHolisitcModel.Holistic(min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

className = input('Write a classname: ')

# Importing the classification model
with open('./bodyLangModel.pkl', 'rb') as file:
    model = pickle.load(file)

capture = cv2.VideoCapture(1)

while True:
    ret, frame = capture.read()

    # Recolor image so media pipe can use it
    recoloredImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = holistic.process(recoloredImage)

    # POSE
    mpDrawingUtils.draw_landmarks(frame, result.pose_landmarks, mpHolisitcModel.POSE_CONNECTIONS)
    # RIGHT HAND
    mpDrawingUtils.draw_landmarks(frame, result.right_hand_landmarks, mpHolisitcModel.HAND_CONNECTIONS)
    # LEFT HAND
    mpDrawingUtils.draw_landmarks(frame, result.left_hand_landmarks, mpHolisitcModel.HAND_CONNECTIONS)
    # FACE
    mpDrawingUtils.draw_landmarks(frame, result.face_landmarks, mpHolisitcModel.FACEMESH_CONTOURS,
                            mpDrawingUtils.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                            mpDrawingUtils.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1) )

    # Exporting coordinates
    try:
        # Extracting pose landmarks
        pose = result.pose_landmarks.landmark
        pose_row = list(numpy.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
        # Extracting face landmarks
        face = result.face_landmarks.landmark
        face_row = list(numpy.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

        # Concat Rows
        row = pose_row + face_row

        # Append class name
        row.insert(0, className)

        # Exporting landmarks to the csv
        with open('coords.csv', mode='a', newline='') as f:
            csvConfig = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvConfig.writerow(row)

    except:
        print('Erro')

    #  to flip the frame
    cv2.imshow('Webcam Feed', cv2.flip(frame,1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()