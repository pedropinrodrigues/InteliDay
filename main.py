import streamlit as st
import cv2
import numpy as np
import os
import json
import time
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

# Interface do Streamlit
tab1, tab2, tab3 = st.tabs(["📸 Cadastrar Pessoa", "🎯 Treinar Modelo", "🔍 Reconhecer"])

# Tab 1: Cadastro
with tab1:
    st.header("Cadastrar Nova Pessoa")
    
    # Reset do formulário se necessário
    if st.session_state.get('clear_form', False):
        st.session_state.clear_form = False
        st.rerun()
    
    person_name = st.text_input("Nome da pessoa:", placeholder="Digite o nome...", key="person_input")
    
    if person_name:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Captura de imagem
            img_file = st.camera_input("Tire uma foto para cadastro")
            
            if img_file is not None:
                # Converter para OpenCV
                image = Image.open(img_file)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Detectar faces
                faces, gray = detect_faces(image_cv)
                
                if len(faces) > 0:
                    st.success(f"✅ {len(faces)} face(s) detectada(s)!")
                    
                    # Mostrar imagem com faces detectadas
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
                            
                            # Reset da interface para novo cadastro
                            st.session_state.clear_form = True
                            time.sleep(2)
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("🔄 Nova Foto", type="secondary", use_container_width=True):
                            st.rerun()
                else:
                    st.warning("⚠️ Nenhuma face detectada! Tente novamente com melhor iluminação.")
        
        with col2:
            # Estatísticas
            dataset_path = Path("face_recognition/dataset")
            if dataset_path.exists():
                people_dirs = [p for p in dataset_path.iterdir() if p.is_dir()]
                st.metric("Pessoas Cadastradas", len(people_dirs))
                
                if people_dirs:
                    st.write("**Lista de pessoas:**")
                    for person_dir in people_dirs:
                        img_count = len(list(person_dir.glob("*.png")))
                        st.write(f"• {person_dir.name}: {img_count} fotos")

# Tab 2: Treinamento
with tab2:
    st.header("Treinar Modelo de Reconhecimento")
    
    if st.button("🚀 Treinar Modelo", type="primary"):
        with st.spinner("Treinando modelo..."):
            success, message = train_model()
            
        if success:
            st.success(message)
            # Limpar cache do recognizer para recarregar
            st.cache_resource.clear()
        else:
            st.error(message)

# Tab 3: Reconhecimento
with tab3:
    st.header("Reconhecimento em Tempo Real")
    
    recognizer, inv_labels = load_recognizer()
    
    if recognizer is None:
        st.warning("⚠️ Modelo não encontrado! Treine o modelo primeiro na aba 'Treinar Modelo'.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            THRESHOLD = st.slider("Limiar de Confiança", 0.0, 150.0, 70.0, 5.0)
            st.caption("Menor valor = mais rigoroso")
            
            # Controles
            start_recognition = st.button("🎥 Iniciar Reconhecimento", type="primary")
            stop_recognition = st.button("⏹️ Parar", type="secondary")
            
            # Placeholder para resultados
            results_placeholder = st.empty()
        
        with col1:
            # Placeholder para vídeo
            video_placeholder = st.empty()
        
        # Estado da aplicação
        if 'recognition_running' not in st.session_state:
            st.session_state.recognition_running = False
        
        if start_recognition:
            st.session_state.recognition_running = True
        
        if stop_recognition:
            st.session_state.recognition_running = False
        
        # Reconhecimento em tempo real
        if st.session_state.recognition_running:
            st.info("🎥 Câmera ativa - Tire fotos continuamente para reconhecimento")
            
            # Câmera simples e estável
            img_file = st.camera_input("📹 Tire uma foto para reconhecer", key="recognition_camera")
            
            if img_file is not None:
                try:
                    # Converter para OpenCV
                    image = Image.open(img_file)
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Detectar faces
                    faces, gray = detect_faces(frame)
                    results = []
                    
                    # Processar cada face detectada
                    for (x, y, w, h) in faces:
                        face = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (200, 200))
                        
                        # Reconhecer face
                        label_id, distance = recognizer.predict(face_resized)
                        name = inv_labels.get(label_id, "Desconhecido")
                        is_known = distance <= THRESHOLD
                        
                        # Definir cores (BGR para OpenCV)
                        if is_known:
                            rect_color = (0, 255, 0)  # Verde
                            bg_color = (0, 200, 0)    # Verde mais escuro
                        else:
                            rect_color = (0, 0, 255)   # Vermelho
                            bg_color = (0, 0, 200)     # Vermelho mais escuro
                        
                        # Desenhar bounding box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 3)
                        
                        # Preparar texto
                        display_name = name if is_known else "DESCONHECIDO"
                        confidence_text = f"{distance:.1f}"
                        
                        # Calcular tamanho do texto
                        (text_width, text_height), _ = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        
                        # Desenhar fundo para o nome
                        cv2.rectangle(frame, (x, y-30), (x + text_width + 10, y), bg_color, -1)
                        
                        # Escrever nome
                        cv2.putText(frame, display_name, (x+5, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Status embaixo
                        status = "✓ AUTORIZADO" if is_known else "✗ NEGADO"
                        cv2.putText(frame, status, (x, y+h+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)
                        
                        results.append({
                            'nome': display_name,
                            'distancia': distance,
                            'conhecido': is_known
                        })
                    
                    # Mostrar resultado
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, 
                                          caption="🔍 Resultado do Reconhecimento", 
                                          use_column_width=True)
                    
                    # Mostrar resultados na lateral
                    with results_placeholder.container():
                        if results:
                            st.write("**📊 Resultados:**")
                            for result in results:
                                if result['conhecido']:
                                    st.success(f"✅ **{result['nome']}** - Confiança: {result['distancia']:.1f}")
                                else:
                                    st.error(f"❌ **Pessoa Desconhecida** - Confiança: {result['distancia']:.1f}")
                        else:
                            st.info("👀 Nenhuma face detectada")
                    
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")
            
            # Auto-refresh para simular continuidade
            if st.button("🔄 Atualizar", key="refresh_recognition"):
                st.rerun()
        
        else:
            video_placeholder.info("📹 Clique em 'Iniciar Reconhecimento' para ativar a câmera")
            results_placeholder.empty()

# Footer
st.markdown("---")
st.markdown("🤖 **Sistema de Reconhecimento Facial LBPH + OpenCV**")
