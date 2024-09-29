import cv2
import mediapipe as mp



webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,500)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()


mp_drawing = mp.solutions.drawing_utils

while True:
    sucess, img = webcam.read()

    if sucess:
        RGB_img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_img)
        
        if result.multi_hand_landmarks:
            for hand_marks in result.multi_hand_landmarks:
                print(hand_marks)
                mp_drawing.draw_landmarks(img, hand_marks,mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand recognition", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
webcam.destroyAllWindows()