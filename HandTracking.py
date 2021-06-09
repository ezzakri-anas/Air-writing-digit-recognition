import mediapipe as mp
import cv2 as cv
import time, math
import numpy as np
import tensorflow as tf


class HandDetector:
    def __init__(self, static_image_mode=False, max_hands=2, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.static_image_mode = static_image_mode
        self.max_hands = max_hands
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_image_mode,
                                         self.max_hands,
                                         self.min_detection_conf,
                                         self.min_tracking_conf)
    
    def hand_detection(self, frame, draw=True):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        self.processing_results = self.hands.process(rgb_frame)
        
        self.multi_hand_landmarks = self.processing_results.multi_hand_landmarks
        self.multi_handedness = self.processing_results.multi_handedness
        if self.multi_hand_landmarks:
            if draw:
                for hand_landmarks in self.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame
            
    def fingersUp(self):
        FINGERS_UP = [0 for _ in range(5)]
        
        if not ((self.Landmarks[4][1] < self.Landmarks[3][1]) ^ self.selected_hand):
            FINGERS_UP[0] = 1
        
        for i in range(4):
            if self.Landmarks[8 + i * 4][2] < self.Landmarks[6 + i * 4][2]:
                FINGERS_UP[1 + i] = 1
        
        return FINGERS_UP
    
    def lm_position(self, frame, preferred_hand, point_to_draw, process_hand=True):
        if process_hand:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            self.processing_results = self.hands.process(rgb_frame)
            self.multi_hand_landmarks = self.processing_results.multi_hand_landmarks
            self.multi_handedness = self.processing_results.multi_handedness
        
        self.Landmarks = []
        
        if self.multi_hand_landmarks:
            
            if len(self.multi_handedness) == 2 and not (self.multi_handedness[0].classification[0].index == 0 ^ preferred_hand): # if "left" is in 1st position then right is second
                mhl = self.multi_hand_landmarks[1]
                self.selected_hand = self.multi_handedness[1].classification[0].index
            else:
                mhl = self.multi_hand_landmarks[0]
                self.selected_hand = self.multi_handedness[0].classification[0].index

            for id, landmark in enumerate(mhl.landmark):
                y, x, _ = frame.shape
                cx = int(landmark.x * x)
                cy = int(landmark.y * y)
                self.Landmarks.append([id, cx, cy])
                
                if id in point_to_draw:
                    cv.circle(frame, (cx, cy), 5, (100, 100, 100), 3)
            
        return self.Landmarks   
          
          
                
# def fps():
#     global ctime
#     ntime = time.time()
#     fps = str(int(1 / (ntime - ctime)))
#     ctime = ntime
#     cv.putText(frame, fps, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def remove_zerows(array):
    new_array = []
    
    while not any(array[0]):
        array = np.delete(array, 0, axis=0)
        if len(array) == 1:
            break
    while not any(array[array.shape[0] - 1]):
        array = np.delete(array, array.shape[0] - 1, axis=0)
        if len(array) < 1:
            break
    return array

def predict_number(canva):
    global Model
    
    canva = cv.GaussianBlur(canva, ksize=(31, 31), sigmaX=0, sigmaY=0)
    canva = cv.cvtColor(canva, cv.COLOR_BGR2GRAY)

    for _ in range(2):
        canva = remove_zerows(canva)
        canva = np.transpose(canva)

    canva = np.pad(canva, ((30, 30), (30, 30)), "constant")
    canva = cv.resize(canva, (28, 28))

    prediction = Model.predict(canva.reshape((-1, 28, 28, 1)))
    res = list(prediction[0])
    max_prediction = max(res)
    return res.index(max_prediction), max_prediction
    

Model = tf.keras.models.load_model("HandWritten_Digits_Model")  

Canvas = np.zeros((480, 640, 3), 'uint8')
prev_point = None
predicted = False

detector = HandDetector(min_detection_conf=0.8, min_tracking_conf=0.85)
cap = cv.VideoCapture(0)

ctime = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame")
        continue
    frame = cv.flip(frame, 1)
    
    # frame = detector.hand_detection(frame, draw=1)
    lms = detector.lm_position(frame, preferred_hand=0, point_to_draw=(8,))
    
    if len(lms):
        isUp = detector.fingersUp()
        
        if ~isUp[0] & isUp[1] & ~isUp[2] & ~isUp[3] & ~isUp[4]:
            predicted = False
            
            curr_point = [lms[8][2], lms[8][1]]
            
            if not prev_point:
                prev_point = curr_point
                
            cv.line(Canvas, (prev_point[1], prev_point[0]), (curr_point[1], curr_point[0]), (1, 199, 160), 14)
            cv.putText(frame, "Ready to write", (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 256), 2)
            prev_point = curr_point
            
        else:
            prev_point = None
            if ~isUp[0] & ~isUp[1] & ~isUp[2] & ~isUp[3] & isUp[4]:
                Canvas = np.zeros((480, 640, 3), 'uint8')
                predicted = False
            
            elif ~isUp[0] & isUp[1] & isUp[2] & ~isUp[3] & ~isUp[4]:
                if not predicted:
                    predicted = True
                    
                    prediction, proba = predict_number(Canvas)
                    if proba > 0.9:
                        message = "I think it is " + str(prediction)
                    elif proba > 0.7:
                        message = "It's probably " + str(prediction)
                    elif proba > 0.5:
                        message = "Is it " + str(prediction) + " ?"
                    else:
                        message = "Never seen this before"
                    
                    
                print(prediction, proba)
                cv.putText(frame, message, (200, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
            cv.putText(frame, "Point Index to write", (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 256), 2)
    
    # fps()
    img = cv.addWeighted(frame, 0.5, Canvas, 0.5, 0)
    cv.imshow("hands_tracking", img)
    
    if cv.waitKey(5) == ord("q"):
      break

cap.release()
cv.destroyAllWindows()
