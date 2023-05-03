from backend.model import *

# Test the model on a live video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Recognize faces and draw bounding boxes with names
    frame = recognize(frame)

    # Display the processed frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()