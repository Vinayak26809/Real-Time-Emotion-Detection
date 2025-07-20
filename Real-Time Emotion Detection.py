import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze the frame for emotions
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        # Draw the emotion label on the frame
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Error:", e)

    # Show the frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()





# 🤖 What It Does
# Captures video from the webcam.

# Detects facial emotions in real-time using DeepFace.

# Displays the dominant emotion (e.g., happy, sad, angry, neutral) over the video feed.

# 📌 Notes
# DeepFace supports multiple backends (opencv, ssd, mtcnn, etc.). The default is fine for most systems.

# If performance is slow, consider reducing the frame resolution or using GPU acceleration.