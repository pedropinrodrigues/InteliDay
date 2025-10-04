# train_lbph.py
import cv2, os, json
from pathlib import Path

DATA_DIR = Path("dataset")
people = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
people.sort()

if not people:
    raise SystemExit("No people found in ./dataset. Run enroll_faces.py first.")

label_map = {name: idx for idx, name in enumerate(people)}
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

images, labels = [], []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for name in people:
    for img_path in (DATA_DIR / name).glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        # (Optional) re-detect to ensure consistent crop; here we trust pre-cropped 200x200
        images.append(img)
        labels.append(label_map[name])

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.train(images, cv2.UMat(labels))  # UMat for compatibility; list also works
recognizer.write("model.yml")
print("[OK] Trained LBPH model saved to model.yml; labels in labels.json")