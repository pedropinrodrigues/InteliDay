# 🔍 InteliDay - Sistema de Reconhecimento Facial

Sistema completo de reconhecimento facial em tempo real usando **LBPH (Local Binary Patterns Histograms)** e **OpenCV**, com interface web construída em **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias](#-tecnologias)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Funciona](#-como-funciona)
- [Configurações](#-configurações)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)

## 🎯 Sobre o Projeto

O **InteliDay** é um sistema de reconhecimento facial desenvolvido para demonstrar conceitos de visão computacional e aprendizado de máquina. O projeto oferece uma interface web intuitiva que permite cadastrar pessoas, treinar modelos de reconhecimento facial e realizar identificações em tempo real.

### Principais Destaques

- ✅ **Interface Web Intuitiva** - Construída com Streamlit
- ✅ **Cadastro Simples** - Tire fotos direto pelo navegador
- ✅ **Treinamento Rápido** - Modelo LBPH com treinamento eficiente
- ✅ **Reconhecimento em Tempo Real** - Janela OpenCV nativa com bounding boxes
- ✅ **Alta Performance** - Processamento otimizado para fluidez
- ✅ **Feedback Visual Rico** - Cores, status e confiança em tempo real

## 🚀 Funcionalidades

### 1. 📸 Cadastro de Pessoas
- Captura de fotos via webcam
- Detecção automática de faces
- Salvamento organizado em dataset
- Reset automático para cadastros múltiplos
- Visualização de estatísticas

### 2. 🎯 Treinamento do Modelo
- Treinamento com algoritmo LBPH
- Suporte a múltiplas pessoas
- Geração automática de labels
- Feedback do processo de treinamento

### 3. 🔍 Reconhecimento em Tempo Real
- Janela OpenCV nativa para máxima performance
- Bounding boxes coloridas (verde/vermelho)
- Nomes e confiança em tempo real
- Informações de timestamp e frames
- Status visual de acesso

## 🛠 Tecnologias

### Core
- **Python 3.8+** - Linguagem principal
- **OpenCV 4.8+** - Visão computacional e processamento de imagem
- **opencv-contrib-python** - Módulo LBPH Face Recognizer

### Interface & Visualização
- **Streamlit 1.28+** - Framework web interativo
- **Pillow** - Processamento de imagens
- **NumPy** - Operações numéricas

### Algoritmos
- **Haar Cascade Classifier** - Detecção de faces
- **LBPH (Local Binary Patterns Histograms)** - Reconhecimento facial

## 📦 Pré-requisitos

- Python 3.8 ou superior
- Webcam funcional
- Sistema operacional: Windows, macOS ou Linux

## 🔧 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/pedropinrodrigues/InteliDay.git
cd InteliDay
```

### 2. Crie um ambiente virtual (recomendado)

```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o aplicativo

```bash
streamlit run main.py
```

O aplicativo abrirá automaticamente no seu navegador em `http://localhost:8501`

## 📖 Como Usar

### Passo 1: Cadastrar Pessoas

1. Acesse a aba **"📸 Cadastrar Pessoa"**
2. Digite o nome da pessoa
3. Posicione-se em frente à câmera
4. Tire uma foto clicando no botão da câmera
5. Clique em **"💾 Salvar Cadastro"**
6. O sistema reinicia automaticamente para novo cadastro

**Dica:** Cadastre a mesma pessoa várias vezes em diferentes ângulos e iluminações para melhor precisão.

### Passo 2: Treinar o Modelo

1. Vá para a aba **"🎯 Treinar Modelo"**
2. Clique em **"🚀 Treinar Modelo"**
3. Aguarde o treinamento completar
4. O modelo será salvo automaticamente

**Importante:** Execute este passo sempre que adicionar novas pessoas ou fotos.

### Passo 3: Reconhecimento em Tempo Real

1. Acesse a aba **"🔍 Reconhecer"**
2. Ajuste o **"Limiar de Confiança"** se necessário (padrão: 70.0)
   - Valores menores = mais rigoroso
   - Valores maiores = mais permissivo
3. Clique em **"🎥 Iniciar Reconhecimento"**
4. Uma janela OpenCV abrirá fora do navegador
5. **Bounding boxes aparecerão automaticamente:**
   - 🟢 **Verde** = Pessoa conhecida (acesso autorizado)
   - 🔴 **Vermelho** = Pessoa desconhecida (acesso negado)
6. Pressione **'q'** na janela da câmera para sair

## 📁 Estrutura do Projeto

```
InteliDay/
│
├── main.py                          # Aplicativo principal Streamlit
├── requirements.txt                 # Dependências do projeto
├── README.md                        # Documentação
│
├── face_recognition/
│   ├── dataset/                     # Imagens cadastradas
│   │   ├── Pessoa1/
│   │   │   ├── 1234567890.png
│   │   │   └── ...
│   │   └── Pessoa2/
│   │       └── ...
│   │
│   ├── model.yml                    # Modelo LBPH treinado
│   ├── labels.json                  # Mapeamento nome → ID
│   │
│   └── utils/                       # Scripts auxiliares (legacy)
│       ├── enroll_faces.py
│       ├── recognize.py
│       ├── real_time.py
│       └── native_view.py
│
└── __pycache__/                     # Cache Python
```

## 🧠 Como Funciona

### 1. Detecção de Faces (Haar Cascade)

O sistema usa o **Haar Cascade Classifier** para detectar faces:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
```

**Como funciona:**
- Converte a imagem para escala de cinza
- Varre a imagem com janela deslizante em múltiplas escalas
- Aplica features Haar em cascata para detectar padrões faciais
- Retorna coordenadas (x, y, largura, altura) das faces detectadas

### 2. Reconhecimento Facial (LBPH)

O **LBPH (Local Binary Patterns Histograms)** analisa padrões locais:

```python
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8
)
recognizer.train(images, labels)
label_id, confidence = recognizer.predict(face)
```

**Funcionamento:**
1. Divide a face em células (grid 8x8)
2. Calcula padrões binários locais em cada pixel
3. Gera histogramas de padrões por célula
4. Compara histogramas com faces treinadas
5. Retorna o ID da pessoa e a distância (confiança)

**Vantagens do LBPH:**
- ✅ Rápido e eficiente
- ✅ Robusto a mudanças de iluminação
- ✅ Não requer GPU
- ✅ Funciona bem com datasets pequenos

### 3. Pipeline de Processamento

```
Câmera → Frame → Escala de Cinza → Detecção → Crop Face → 
Resize (200x200) → LBPH Predict → Comparar com Threshold → 
Desenhar Bounding Box → Mostrar Resultado
```

## ⚙️ Configurações

### Ajustar Threshold de Confiança

O threshold determina quão rigoroso é o reconhecimento:

```python
THRESHOLD = 70.0  # Valor padrão
```

- **Menor (ex: 50):** Mais rigoroso, menos falsos positivos, pode rejeitar conhecidos
- **Maior (ex: 100):** Mais permissivo, aceita mais pessoas, mais falsos positivos

### Parâmetros do Haar Cascade

```python
faces = face_cascade.detectMultiScale(
    gray,           # Imagem em escala de cinza
    scaleFactor=1.1,  # Redução de escala (1.05-1.3)
    minNeighbors=5,   # Detecções mínimas para confirmar (3-6)
    minSize=(80, 80)  # Tamanho mínimo da face
)
```

### Configuração da Câmera

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

## 📊 Parâmetros de Performance

| Parâmetro | Valor Padrão | Impacto |
|-----------|--------------|---------|
| Resolução | 640x480 | Performance vs Qualidade |
| Threshold | 70.0 | Precisão do reconhecimento |
| scaleFactor | 1.1 | Velocidade de detecção |
| minNeighbors | 5 | Falsos positivos |
| Face Size | 200x200 | Tamanho normalizado |

## 🔍 Troubleshooting

### Câmera não detectada
```bash
# Verificar permissões da câmera no sistema
# macOS: Preferências do Sistema → Segurança → Câmera
# Windows: Configurações → Privacidade → Câmera
```

### Modelo não reconhece faces
- **Solução 1:** Cadastre mais fotos da mesma pessoa (5-10 fotos)
- **Solução 2:** Tire fotos em diferentes iluminações e ângulos
- **Solução 3:** Aumente o threshold (ex: 80-100)
- **Solução 4:** Retreine o modelo após adicionar mais fotos

### Performance lenta
- **Solução 1:** Reduza a resolução da câmera
- **Solução 2:** Aumente o `scaleFactor` (ex: 1.2)
- **Solução 3:** Aumente o `minSize` (ex: 100x100)

### Muitos falsos positivos
- **Solução 1:** Reduza o threshold (ex: 50-60)
- **Solução 2:** Aumente o `minNeighbors` (ex: 6-8)
- **Solução 3:** Melhore a iluminação do ambiente

## 🎨 Personalização

### Alterar Cores das Bounding Boxes

No arquivo `main.py`:

```python
# Pessoa conhecida
rect_color = (0, 255, 0)  # BGR: Verde
bg_color = (0, 200, 0)

# Pessoa desconhecida
rect_color = (0, 0, 255)  # BGR: Vermelho
bg_color = (0, 0, 200)
```

### Adicionar Novos Textos na Tela

```python
cv2.putText(frame, "Seu Texto", (x, y), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Roadmap

- [ ] Suporte a múltiplas câmeras
- [ ] Exportação de logs de acesso
- [ ] Dashboard de estatísticas
- [ ] Integração com banco de dados
- [ ] API REST para integração
- [ ] Suporte a modelos DNN (MTCNN, RetinaFace)
- [ ] Detecção de máscara facial
- [ ] Reconhecimento de emoções

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

**Pedro Pinheiro Rodrigues**

- GitHub: [@pedropinrodrigues](https://github.com/pedropinrodrigues)

## 📚 Referências

- [OpenCV Face Recognition Documentation](https://docs.opencv.org/master/dd/d65/classcv_1_1face_1_1FaceRecognizer.html)
- [LBPH Algorithm](https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b)
- [Haar Cascade Classifiers](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---