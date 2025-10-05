import streamlit as st
import cv2
import numpy as np
import os
import json
import time
from pathlib import Path
from PIL import Image
import av
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# Configuração da página
st.set_page_config(page_title="Sistema de Reconhecimento Facial", layout="wide")
st.title("🔍 Sistema de Reconhecimento Facial")

# Inicializar classificador de faces
@st.cache_resource
def load_face_classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_cascade = load_face_classifier()

# Estado global para reconhecimento
if 'webrtc_playing' not in st.session_state:
    st.session_state.webrtc_playing = False

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

# Componentes de interface reutilizáveis
TAB_LABELS = ["📸 Cadastrar Pessoa", "🎯 Treinar Modelo", "🔍 Reconhecer"]

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self, recognizer, inv_labels, threshold):
        self.recognizer = recognizer
        self.inv_labels = inv_labels
        self.threshold = threshold

    def set_threshold(self, threshold):
        self.threshold = threshold

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        faces, gray = detect_faces(image)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))

            label_id, distance = self.recognizer.predict(face_resized)
            name = self.inv_labels.get(label_id, "Desconhecido")
            is_known = distance <= self.threshold

            rect_color = (0, 255, 0) if is_known else (0, 0, 255)
            bg_color = (0, 200, 0) if is_known else (0, 0, 200)

            cv2.rectangle(image, (x, y), (x+w, y+h), rect_color, 3)

            display_name = name if is_known else "DESCONHECIDO"
            (text_width, _), _ = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(image, (x, y-30), (x + text_width + 10, y), bg_color, -1)
            cv2.putText(image, display_name, (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            status = "✓ AUTORIZADO" if is_known else "✗ NEGADO"
            cv2.putText(image, status, (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

def render_register_tab():
    st.header("Cadastrar Nova Pessoa")

    if st.session_state.get('clear_form', False):
        st.session_state.clear_form = False
        st.rerun()

    person_name = st.text_input("Nome da pessoa:", placeholder="Digite o nome...", key="person_input")

    if person_name:
        col1, col2 = st.columns([2, 1])

        with col1:
            if st.session_state.webrtc_playing:
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
                            if st.button("💾 Salvar Cadastro", type="primary", width="stretch"):
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
                            if st.button("🔄 Nova Foto", type="secondary", width="stretch"):
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

    threshold = st.slider("Limiar de Confiança", 0.0, 150.0, 70.0, 5.0)
    st.caption("Menor valor = mais rigoroso")

    def processor_factory():
        return FaceRecognitionProcessor(recognizer, inv_labels, threshold)

    ctx = webrtc_streamer(
        key="face-recognition",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=processor_factory,
        async_processing=False,
        video_html_attrs={
            "style": {
                "width": "1440px",
                "maxWidth": "100%",
                "height": "1080px",
                "borderRadius": "12px",
                "objectFit": "cover",
            },
            "playsInline": True,
            "controls": False,
            "autoPlay": True,
        },
    )

    if ctx.state.playing:
        st.session_state.webrtc_playing = True
        if ctx.video_processor:
            ctx.video_processor.set_threshold(threshold)
    else:
        st.session_state.webrtc_playing = False
        st.info("📹 Clique em 'Start' para iniciar o reconhecimento em tempo real")


if 'active_tab' not in st.session_state:
    st.session_state.active_tab = TAB_LABELS[0]

selected_tab = st.radio("Navegação", TAB_LABELS,
                        index=TAB_LABELS.index(st.session_state.active_tab),
                        horizontal=True, label_visibility="collapsed")
st.session_state.active_tab = selected_tab

if selected_tab != TAB_LABELS[2]:
    st.session_state.webrtc_playing = False

if selected_tab == TAB_LABELS[0]:
    render_register_tab()
elif selected_tab == TAB_LABELS[1]:
    render_training_tab()
else:
    render_recognition_tab()

# Footer
st.markdown("---")
st.markdown("🤖 **Sistema de Reconhecimento Facial LBPH + OpenCV**")
