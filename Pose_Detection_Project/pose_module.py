import time
import mediapipe as mp
import cv2

class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectioncon = detectioncon
        self.trackcon = trackcon  

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody,
                                     self.smooth, self.detectioncon,
                                     self.trackcon)  
    
    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(imgRGB)

        if result.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, result.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img  

def main():
    cap = cv2.VideoCapture("video2.mp4")

    if not cap.isOpened():
        print("Error: Could not open video file. Check the file path and format.")
        return  

    previous_time = 0
    detector = PoseDetector()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Error: Failed to read video frame or video ended.")
            break

        img = cv2.resize(img, (640, 480))  # Resize for better performance
        img = detector.find_pose(img)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, f'FPS: {int(fps)}', (40, 60), cv2.FONT_HERSHEY_PLAIN, 
                    3, (45, 67, 89), 3)  

        cv2.imshow('Pose Detection', img)

        if cv2.waitKey(30) & 0xFF == ord("q"):  # Delay added for smooth playback
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
