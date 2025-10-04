# recognize_lbph.py
import cv2, json
import numpy as np

# Load model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")
with open("labels.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
inv_labels = {v: k for k, v in label_map.items()}

THRESHOLD = 70.0  # lower is better; adjust after testing
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label_id, distance = recognizer.predict(face)
        name = inv_labels.get(label_id, "Unknown")
        is_known = distance <= THRESHOLD

        color = (0, 200, 0) if is_known else (0, 0, 255)
        status = "ACCESS GRANTED" if is_known else "ACCESS DENIED"
        shown = name if is_known else "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{shown} ({distance:.1f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, status, (x, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition (LBPH)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()