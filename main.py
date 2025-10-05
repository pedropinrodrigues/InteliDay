import streamlit as st
import cv2
import numpy as np
import os
import json
import time
import subprocess
import sys
from pathlib import Path
from PIL import Image

# Configuração da página
st.set_page_config(page_title="Sistema de Reconhecimento Facial", layout="wide")
st.title("🔍 Sistema de Reconhecimento Facial")

# Inicializar classificador de faces
@st.cache_resource
def load_face_classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_cascade = load_face_classifier()

# Estado global para a janela nativa de reconhecimento
if 'native_camera_process' not in st.session_state:
    st.session_state.native_camera_process = None
if 'native_camera_threshold' not in st.session_state:
    st.session_state.native_camera_threshold = 70.0
if 'native_camera_error' not in st.session_state:
    st.session_state.native_camera_error = None

# Funções auxiliares
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    return faces, gray

def save_face_image(face_image, person_name):
    save_dir = os.path.join("face_recognition", "dataset", person_name)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{int(time.time()*1000)}.png")
    cv2.imwrite(filename, face_image)
    return filename

def train_model():
    DATA_DIR = Path("face_recognition/dataset")
    people = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
    
    if not people:
        return False, "Nenhuma pessoa cadastrada!"
    
    people.sort()
    label_map = {name: idx for idx, name in enumerate(people)}
    
    # Salvar labels
    with open("face_recognition/labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    images, labels = [], []
    
    for name in people:
        for img_path in (DATA_DIR / name).glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label_map[name])
    
    if not images:
        return False, "Nenhuma imagem encontrada!"
    
    # Treinar modelo
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    recognizer.write("face_recognition/model.yml")
    
    return True, f"Modelo treinado com {len(images)} imagens de {len(people)} pessoas!"

@st.cache_resource
def load_recognizer():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("face_recognition/model.yml")
        with open("face_recognition/labels.json", "r", encoding="utf-8") as f:
            label_map = json.load(f)
        inv_labels = {v: k for k, v in label_map.items()}
        return recognizer, inv_labels
    except:
        return None, None


def annotate_frame(frame_bgr, recognizer, inv_labels, threshold):
    annotated = frame_bgr.copy()
    faces, gray = detect_faces(frame_bgr)

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


# Componentes de interface reutilizáveis
TAB_LABELS = ["📸 Cadastrar Pessoa", "🎯 Treinar Modelo", "🔍 Reconhecer"]

def render_register_tab():
    st.header("Cadastrar Nova Pessoa")

    if st.session_state.get('clear_form', False):
        st.session_state.clear_form = False
        st.rerun()

    person_name = st.text_input("Nome da pessoa:", placeholder="Digite o nome...", key="person_input")

    if person_name:
        col1, col2 = st.columns([2, 1])

        with col1:
            proc = st.session_state.native_camera_process
            native_running = proc is not None and proc.poll() is None
            if native_running:
                st.info("⚠️ Pare o reconhecimento em tempo real antes de cadastrar uma nova pessoa.")
            else:
                img_file = st.camera_input("Tire uma foto para cadastro")

                if img_file is not None:
                    image = Image.open(img_file)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    faces, gray = detect_faces(image_cv)

                    if len(faces) > 0:
                        st.success(f"✅ {len(faces)} face(s) detectada(s)!")

                        for (x, y, w, h) in faces:
                            cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption="Faces detectadas")

                        col_btn1, col_btn2 = st.columns([1, 1])

                        with col_btn1:
                            if st.button("💾 Salvar Cadastro", type="primary", use_container_width=True):
                                saved_count = 0
                                for (x, y, w, h) in faces:
                                    face = gray[y:y+h, x:x+w]
                                    face_resized = cv2.resize(face, (200, 200))
                                    save_face_image(face_resized, person_name)
                                    saved_count += 1

                                st.success(f"🎉 {saved_count} imagem(ns) salva(s) para {person_name}!")
                                st.balloons()

                                st.session_state.clear_form = True
                                time.sleep(2)
                                st.rerun()

                        with col_btn2:
                            if st.button("🔄 Nova Foto", type="secondary", use_container_width=True):
                                st.rerun()
                    else:
                        st.warning("⚠️ Nenhuma face detectada! Tente novamente com melhor iluminação.")

        with col2:
            dataset_path = Path("face_recognition/dataset")
            if dataset_path.exists():
                people_dirs = [p for p in dataset_path.iterdir() if p.is_dir()]
                st.metric("Pessoas Cadastradas", len(people_dirs))

                if people_dirs:
                    st.write("**Lista de pessoas:**")
                    for person_dir in people_dirs:
                        img_count = len(list(person_dir.glob("*.png")))
                        st.write(f"• {person_dir.name}: {img_count} fotos")

def render_training_tab():
    st.header("Treinar Modelo de Reconhecimento")

    col_train, col_reset = st.columns([2, 1])

    with col_train:
        if st.button("🚀 Treinar Modelo", type="primary", use_container_width=True):
            with st.spinner("Treinando modelo..."):
                success, message = train_model()

            if success:
                st.success(message)
                st.cache_resource.clear()
            else:
                st.error(message)

    with col_reset:
        if st.button("🗑️ Resetar Modelo", type="secondary", use_container_width=True):
            model_path = Path("face_recognition/model.yml")
            labels_path = Path("face_recognition/labels.json")

            deleted = []
            for path in [model_path, labels_path]:
                try:
                    path.unlink(missing_ok=True)
                    deleted.append(path.name)
                except Exception as exc:
                    st.error(f"Erro ao remover {path.name}: {exc}")
                    return

            st.cache_resource.clear()
            if deleted:
                st.success("Arquivos redefinidos: " + ", ".join(deleted))
            else:
                st.info("Nenhum arquivo de modelo encontrado para remover.")

    if st.button("🚀 Treinar Modelo", type="primary"):
        with st.spinner("Treinando modelo..."):
            success, message = train_model()

        if success:
            st.success(message)
            st.cache_resource.clear()
        else:
            st.error(message)

def render_recognition_tab():
    st.header("Reconhecimento em Tempo Real")

    recognizer, inv_labels = load_recognizer()

    if recognizer is None:
        st.warning("⚠️ Modelo não encontrado! Treine o modelo primeiro na aba 'Treinar Modelo'.")
        return

    threshold = st.slider(
        "Limiar de Confiança",
        0.0,
        150.0,
        float(st.session_state.native_camera_threshold),
        5.0,
    )
    st.session_state.native_camera_threshold = threshold
    st.caption("Menor valor = mais rigoroso")

    proc = st.session_state.native_camera_process
    running = proc is not None and proc.poll() is None
    if running:
        st.caption("Para aplicar um novo limiar, feche e reabra a janela nativa.")

    col_start, col_stop = st.columns(2)

    proc = st.session_state.native_camera_process
    running = proc is not None and proc.poll() is None

    with col_start:
        start_native = st.button(
            "🎥 Abrir janela nativa",
            type="primary",
            use_container_width=True,
            disabled=running,
        )
    with col_stop:
        stop_native = st.button(
            "⏹️ Fechar janela nativa",
            type="secondary",
            use_container_width=True,
            disabled=not running,
        )

    proc = st.session_state.native_camera_process
    running = proc is not None and proc.poll() is None

    if start_native:
        if running:
            st.info("A janela nativa já está aberta. Feche-a manualmente ou use 'Fechar janela nativa'.")
        else:
            script_path = Path(__file__).resolve().parent / "face_recognition" / "utils" / "native_view.py"
            if not script_path.exists():
                st.error("Arquivo 'face_recognition/utils/native_view.py' não encontrado.")
            else:
                try:
                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--threshold",
                        str(threshold),
                    ]
                    env = os.environ.copy()
                    proc = subprocess.Popen(cmd, env=env)
                    time.sleep(0.2)
                    if proc.poll() is not None and proc.returncode not in (0, None):
                        st.session_state.native_camera_process = None
                        st.session_state.native_camera_error = (
                            f"Processo da câmera finalizou imediatamente (código {proc.returncode})."
                        )
                    else:
                        st.session_state.native_camera_process = proc
                        running = True
                        st.success("Janela nativa aberta. Pressione 'q' nela ou use 'Fechar'.")
                except Exception as exc:
                    st.session_state.native_camera_error = str(exc)
                    st.error(f"Erro ao abrir janela nativa: {exc}")

    if stop_native and running:
        proc = st.session_state.native_camera_process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        st.session_state.native_camera_process = None
        st.session_state.native_camera_error = None
        running = False

    proc = st.session_state.native_camera_process
    if proc is not None and proc.poll() is not None:
        return_code = proc.returncode
        st.session_state.native_camera_process = None
        running = False
        if return_code not in (0, None):
            st.session_state.native_camera_error = (
                st.session_state.native_camera_error
                or f"Processo da câmera terminou com código {return_code}."
            )

    error_message = st.session_state.native_camera_error
    if error_message:
        st.error(error_message)
        st.session_state.native_camera_error = None

    if running:
        st.info("Janela nativa em execução. Pressione 'q' na janela ou use o botão de fechar para encerrar.")
    else:
        st.caption("A captura acontece fora do navegador. Pressione 'q' na janela nativa para fechá-la.")


if 'active_tab' not in st.session_state:
    st.session_state.active_tab = TAB_LABELS[0]

selected_tab = st.radio("Navegação", TAB_LABELS,
                        index=TAB_LABELS.index(st.session_state.active_tab),
                        horizontal=True, label_visibility="collapsed")
st.session_state.active_tab = selected_tab

if selected_tab != TAB_LABELS[2]:
    proc = st.session_state.native_camera_process
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    st.session_state.native_camera_process = None

if selected_tab == TAB_LABELS[0]:
    render_register_tab()
elif selected_tab == TAB_LABELS[1]:
    render_training_tab()
else:
    render_recognition_tab()

# Footer
st.markdown("---")
st.markdown("🤖 **Sistema de Reconhecimento Facial LBPH + OpenCV**")
