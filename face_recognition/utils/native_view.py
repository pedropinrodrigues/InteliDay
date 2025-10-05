#!/usr/bin/env python3
"""Abrir janela nativa para reconhecimento facial usando LBPH."""

import argparse
import json
import sys
from pathlib import Path

import cv2


def load_recognizer(model_path: Path, labels_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Arquivo de labels não encontrado: {labels_path}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))

    with labels_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    inv_labels = {int(idx): name for name, idx in label_map.items()}
    return recognizer, inv_labels


def load_cascade():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError("Não foi possível carregar o classificador Haarcascade.")
    return cascade


def annotate_frame(frame_bgr, recognizer, inv_labels, threshold, cascade):
    annotated = frame_bgr.copy()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    if len(faces) == 0:
        cv2.putText(annotated, "Nenhuma face detectada", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return annotated

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200, 200))
        label_id, distance = recognizer.predict(face_resized)
        name = inv_labels.get(label_id, "Desconhecido")
        is_known = distance <= threshold

        rect_color = (0, 255, 0) if is_known else (0, 0, 255)
        bg_color = (0, 200, 0) if is_known else (0, 0, 200)

        cv2.rectangle(annotated, (x, y), (x+w, y+h), rect_color, 3)

        display_name = name if is_known else "DESCONHECIDO"
        (text_width, _), _ = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x, y-30), (x + text_width + 10, y), bg_color, -1)
        cv2.putText(annotated, display_name, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        status = "✓ AUTORIZADO" if is_known else "✗ NEGADO"
        cv2.putText(annotated, status, (x, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)

    return annotated


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconhecimento facial em janela nativa.")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "model.yml"),
                        help="Caminho para o arquivo do modelo treinado (YML).")
    parser.add_argument("--labels", default=str(Path(__file__).resolve().parents[1] / "labels.json"),
                        help="Caminho para o arquivo de labels (JSON).")
    parser.add_argument("--threshold", type=float, default=70.0,
                        help="Limiar de confiança (menor = mais rigoroso).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Índice da webcam (0 = padrão).")
    args = parser.parse_args(argv)

    recognizer, inv_labels = load_recognizer(Path(args.model), Path(args.labels))
    cascade = load_cascade()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a webcam do dispositivo.", file=sys.stderr)
        return 1

    window_name = "Reconhecimento Facial - Janela Nativa"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    print("[INFO] Pressione 'q' para encerrar a janela.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERRO] Falha ao ler frames da webcam.", file=sys.stderr)
                break

            annotated = annotate_frame(frame, recognizer, inv_labels, args.threshold, cascade)
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
