import streamlit as st
import pickle
import pyttsx3
import cv2
import mediapipe as mp
import numpy as np

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
engine.setProperty("rate",170)
model_dict = pickle.load(open('C:\\Users\\krish\\OneDrive\\Documents\\sign_language_technoxian (2) (1)\\sign_language_technoxian\\model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

count = 0
predictedString = ''

labels_dict = {0: 'Hello', 1: 'I LOVE YOU', 2: 'A',3:'B',4:'C',5:'D',6:'H',7:'O',8:'W',9:'I',10:'K',11:'R',12:'U',13:'YES',14:'NO',15:'Y'}


st.title("Tech Titans")

frame_placeholder=st.empty()
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame=cv2.resize(frame,(0,0),None,0.670,0.670)

    frame = frame[150:690,45:690]

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]


 


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        
        if count < 2 and predictedString == predicted_character:
            engine.say(predicted_character)
            engine.runAndWait()
        elif predictedString != predicted_character:
            count = 0

        predictedString = predicted_character     
        count += 1

    frame_placeholder.image(frame,channels="RGB")


    #cv2.imshow('frame', frame)
    #cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()