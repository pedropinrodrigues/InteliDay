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
    
    person_name = st.text_input("Nome da pessoa:", placeholder="Digite o nome...")
    
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
                    
                    if st.button("💾 Salvar Cadastro", type="primary"):
                        saved_count = 0
                        for (x, y, w, h) in faces:
                            face = gray[y:y+h, x:x+w]
                            face_resized = cv2.resize(face, (200, 200))
                            save_face_image(face_resized, person_name)
                            saved_count += 1
                        
                        st.success(f"🎉 {saved_count} imagem(ns) salva(s) para {person_name}!")
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
        THRESHOLD = st.slider("Limiar de Confiança", 0.0, 150.0, 70.0, 5.0)
        st.caption("Menor valor = mais rigoroso")
        
        # Captura para reconhecimento
        img_file = st.camera_input("Tire uma foto para reconhecimento")
        
        if img_file is not None:
            # Converter para OpenCV
            image = Image.open(img_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detectar e reconhecer faces
            faces, gray = detect_faces(image_cv)
            
            if len(faces) > 0:
                results = []
                
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (200, 200))
                    
                    # Reconhecer
                    label_id, distance = recognizer.predict(face_resized)
                    name = inv_labels.get(label_id, "Desconhecido")
                    is_known = distance <= THRESHOLD
                    
                    # Desenhar resultado
                    color = (0, 255, 0) if is_known else (0, 0, 255)
                    status = "✅ ACESSO LIBERADO" if is_known else "❌ ACESSO NEGADO"
                    shown_name = name if is_known else "Desconhecido"
                    
                    cv2.rectangle(image_cv, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(image_cv, f"{shown_name} ({distance:.1f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    results.append({
                        'nome': shown_name,
                        'distancia': distance,
                        'status': status,
                        'conhecido': is_known
                    })
                
                # Mostrar resultado
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption="Resultado do Reconhecimento")
                
                with col2:
                    st.write("**Resultados:**")
                    for i, result in enumerate(results, 1):
                        st.write(f"**Face {i}:**")
                        st.write(f"Nome: {result['nome']}")
                        st.write(f"Confiança: {result['distancia']:.1f}")
                        st.write(result['status'])
                        st.write("---")
            else:
                st.warning("⚠️ Nenhuma face detectada!")

# Footer
st.markdown("---")
st.markdown("🤖 **Sistema de Reconhecimento Facial LBPH + OpenCV**")
