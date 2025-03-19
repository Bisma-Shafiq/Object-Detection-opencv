import time
import mediapipe as mp
import cv2
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


cap = cv2.VideoCapture('video2.mp4')
previous_time =0
while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    print(result.pose_landmarks)
    if result.pose_landmarks:     # dot with connection
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # axis at every pose point
        for id , lm in enumerate(result.pose_landmarks.landmark):
            h , w , c = img.shape
            print(id, lm)
            # cicle across poits
            cx , cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img , (cx,cy),5,(244,23,45),cv2.FILLED)




    curret_time = time.time()
    fps = 1/(curret_time-previous_time)
    previous_time = curret_time 
    cv2.putText(img , str(int(fps)),(40,60),cv2.FONT_HERSHEY_PLAIN,3,(45,67,89),3)   # print fps on video
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()