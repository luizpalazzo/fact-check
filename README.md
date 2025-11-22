# 🎬 Cortes Inteligentes com IA  
### Pipeline Completo para gerar Shorts automatizados com IA (YouTube + Reels)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Enabled-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange.svg)
![Whisper](https://img.shields.io/badge/Whisper-Offline-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)

Este projeto é um **pipeline 100% automatizado** capaz de transformar vídeos longos (podcasts, entrevistas etc.) em **shorts virais e inteligentes**, com:

- 🎯 Cortes gerados por IA (GPT-4o)  
- 🧠 Fact-check com veredito + fonte  
- 🎙️ Narração natural via ElevenLabs  
- 🎨 Versões 16:9 e 9:16 geradas automaticamente  
- 🧩 Edição completa via FFmpeg  
- 📤 Upload automático para YouTube e Facebook Reels  
- 📊 Relatório final de custos (OpenAI + ElevenLabs)

---

# 📌 Índice
- [Funcionalidades](#-funcionalidades)
- [Tecnologias](#-tecnologias)
- [Onde colocar suas Keys](#-onde-colocar-suas-keys)
- [Como Rodar Localmente](#-como-rodar-localmente)

---

# 🔥 Funcionalidades

## 🧠 Inteligência Artificial
- Transcrição offline com **Whisper large-v3-turbo**
- Diarização automática com **Resemblyzer + Spectral Clustering**
- Sugestões de cortes usando GPT-4o com base no conteúdo real
- Fact-check com:
  - Classificação (Verdadeiro, Falso, Impreciso, Relato)
  - Explicação resumida
  - Fonte confiável
  - Narração automática

## 🎙️ Narração & Vídeo
- Narração natural via ElevenLabs com fallback automático
- Vinheta automática com logo e áudio
- Overlays gráficos (logo + ícone)
- Fade-out automático
- Shorts 16:9 e 9:16 (com fundo blur)
- Card visual de veredito ao final

## 📤 Publicação Automática
- YouTube:
  - Título otimizado
  - Thumbnail automática
  - Descrição SEO-friendly
  - Tags e hashtags
- Facebook Reels (vertical)

## 📊 Operacional
- Cálculo de custo OpenAI + ElevenLabs
- Logs detalhados
- Tratamento de erros
- Cache local

---

# 🧩 Tecnologias

| Categoria | Ferramentas |
|----------|-------------|
| IA | OpenAI GPT-4o, Whisper Offline |
| TTS | ElevenLabs |
| Vídeo | FFmpeg / FFprobe |
| Download | yt-dlp |
| Diarização | Resemblyzer, SpectralCluster |
| Backend | Python 3.10+ |
| Uploads | YouTube API, Facebook Graph API |

---

# 🔐 Onde colocar suas Keys

No início do arquivo `main.py`:

```python
YOUTUBE_URL = "https://www.youtube.com/watch?v=XXXX"

OPENAI_API_KEY = "SUA_OPENAI_KEY"

ELEVEN_API_KEYS = [
    "SUA_ELEVEN_KEY_1",
    "SUA_ELEVEN_KEY_2"
]

FACEBOOK_PAGE_ID = "ID_DA_PAGINA"
FACEBOOK_ACCESS_TOKEN = "TOKEN_FACEBOOK"
```

---

# ▶️ Como Rodar Localmente

### 1. Instale as dependências
```
pip install -r requirements.txt
```

### 2. Instale o FFmpeg

**Windows:**  
Baixe em: https://www.gyan.dev/ffmpeg/builds/  
Adicione a pasta `bin/` ao PATH.

**Linux (Ubuntu/Debian):**
```
sudo apt update && sudo apt install ffmpeg
```

**MacOS (Homebrew):**
```
brew install ffmpeg
```

---

### 3. Configure suas chaves no `main.py`

```python
OPENAI_API_KEY = "SUA_OPENAI_KEY"
ELEVEN_API_KEYS = ["SUA_KEY1", "SUA_KEY2"]
FACEBOOK_PAGE_ID = "ID_DA_PAGINA"
FACEBOOK_ACCESS_TOKEN = "TOKEN_FACEBOOK"
YOUTUBE_URL = "https://www.youtube.com/watch?v=XXXXXXXX"
```

---

### 4. (Opcional) Usar cookies do YouTube

Crie `cookies.txt` na raiz e habilite em `main.py`:

```python
USE_COOKIES = True
```

---

### 5. Execute o script

```
python main.py
```

---

### 6. O pipeline executa automaticamente:
- Download do vídeo  
- Transcrição (Whisper offline)  
- Diarização  
- Sugestão de cortes (GPT-4o)  
- Fact-check  
- Narração (ElevenLabs)  
- Edição via FFmpeg  
- Shorts 16:9  
- Shorts 9:16 vertical  
- Vídeo longo final  
- Upload YouTube  
- Upload Facebook Reels  
- Relatório final de custo  

---

### 7. Arquivos de saída
```
shorts_final/
    short_01_final.mp4
    short_01_final_vertical.mp4

long_final/
    video_padrao_final.mp4
    thumbnail.png
```
