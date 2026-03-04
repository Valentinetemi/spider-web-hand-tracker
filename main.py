import cv2 as cv
import mediapipe as mp



#Initialize mediapipe hands
mp_hands = mp.solutions.hands #this import the hand tracking model
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) #hand create the hand detector/ tracker instance
mp_draw = mp.solutions.drawing_utils #this is lets you draw hand landmark on the video frame.

#start the web camera

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret :
        break
    
    frame = cv.flip(frame, 1)
    
    #convert bgr to rgb
    rbg_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    #process the frame to find the hands
    result = hands.process(rbg_frame)
    
    #if the hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            #draw the hand land markers from the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            #get the hand landmarks
            h, w, _ = frame.shape
            points = []
            
            for lm in hand_landmarks.landmark:
                x,y = int(lm.x * w), int(lm.y * h)
                points.append((x,y))
                
            #draw the spider-web on the frame(connect all the point to each there)
            for i in range(0, len(points), 3): # thicker web spacing
                for j in range(i + 2, len(points), 3): #connect every 2nd point to the next 2ndpoint
                    #generate color for the line based on the distance between the points
                    #crystal gradient color (glass-like effect)
                    r =int(128 + 127 * (i / len(points)))
                    g = int(200 + 55 * ((i + j ) % 3))
                    b = int(255 * (j / len(points)))
                    color = (b % 255, g % 255, r % 255)
                    
                    cv.line(frame, points[i], points[j], color, 2)
        else:
            None
            
    #show webcam
    cv.imshow("Spider-Web hand tracker", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()



