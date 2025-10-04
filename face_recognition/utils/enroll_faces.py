# enroll_faces.py
import cv2, os, time, sys

person = sys.argv[1] if len(sys.argv) > 1 else input("Person name: ").strip()
save_dir = os.path.join("dataset", person)
os.makedirs(save_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

count, target = 0, 30  # ~30 samples
print(f"[INFO] Enrolling {person}. Look at the camera. Press 'q' to quit early.")

while True:
    ok, frame = cap.read()
    if not ok: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):  # Press 's' to start capturing
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))  # normalize size
            filename = os.path.join(save_dir, f"{int(time.time()*1000)}.png")
            cv2.imwrite(filename, face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Saved {count}/{target}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            time.sleep(0.05)
    else:
        # Draw rectangles around detected faces even when not capturing
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("Enroll", frame)
    if key == ord('q') or count >= target:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[OK] Collected {count} images for {person} in {save_dir}")