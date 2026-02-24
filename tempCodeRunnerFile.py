import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("cabbage2.pt")

cap = cv2.VideoCapture(1)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model.track(frame, persist=True, device="cpu")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                x, y, w, h = box.xywh[0]
                cx = int(x)
                cy = int(y)
                
                cv2.circle(
                    annotated_frame,
                    (cx, cy),
                    radius = 5,
                    color = (0, 0, 255),
                    thickness = -1
                )
                
        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()