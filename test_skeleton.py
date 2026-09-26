import cv2
from ultralytics import YOLO

def main():
    print("Loading YOLO Pose Model...")
    model = YOLO("models/yolo26n-pose.pt")
    
    print("Opening Webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened! Press 'q' in the video window to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run YOLO pose inference
        results = model(frame, verbose=False)
        
        # Let YOLO natively draw the beautiful green skeleton and red dots!
        annotated_frame = results[0].plot()
        
        # Display it
        cv2.imshow("YOLO Full Skeleton Test (Press 'q' to quit)", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
