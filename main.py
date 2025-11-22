# -*- coding: utf-8 -*-
"""
main_v15_full.py — shorts (16:9 e 9:16) + vídeo padrão (16:9) com intro narrada + thumbnail
- Shorts 9:16 sem crop (mantém todo conteúdo): pad + fundo blur (estilo TikTok/YT)
- Vídeo padrão 16:9: intro narrada via GPT + ElevenLabs, 3 cortes, veredito após cada corte
- Limpeza no final: mantém apenas finais (shorts, versão vertical, vídeo padrão e thumbnail)

Requisitos:
- Python 3.10+
- FFmpeg e FFprobe no PATH
- pip install: yt-dlp faster-whisper resemblyzer spectralcluster numpy librosa pydub openai elevenlabs requests sentencepiece Cython
"""

import os
import sys
import re
import json
import shutil
import subprocess
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request


# ======================== DEPENDÊNCIAS DE ÁUDIO/ML ========================
import numpy as np
from faster_whisper import WhisperModel

# Diarização local
from resemblyzer import VoiceEncoder
from spectralcluster import SpectralClusterer

# TTS (ElevenLabs SDK oficial)
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# OpenAI
from openai import OpenAI as OpenAIChat
from openai import OpenAI as OpenAIImages
import base64

import random


# ============================== CONFIG GERAL ===============================
# >>>> SUBSTITUA SUAS KEYS AQUI <<<<
YOUTUBE_URL     = "https://www.youtube.com/watch?v=xxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxx"  # insira sua key localmente

ELEVEN_API_KEYS = [
    "xxxxxxx",
    "xxxxxxx"
]

# Credenciais Facebook
FACEBOOK_PAGE_ID = "123456"  # substitua pelo ID da página
FACEBOOK_ACCESS_TOKEN = "xxxxxxxxxx"      # token de acesso da página


# === MÉTRICAS DE CUSTO ===
openai_stats = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
eleven_stats = {"requests": 0, "characters": 0}



# Pastas
ROOT_DIR   = Path(".").resolve()
ASSETS_DIR = ROOT_DIR / "assets"
OUT_DIR    = ROOT_DIR / "saida"         # Intermediários (será limpado ao final)
SHORTS_DIR = ROOT_DIR / "shorts_final"  # Finais dos shorts
LONG_DIR   = ROOT_DIR / "long_final"    # Final do vídeo padrão + thumbnail

# Arquivos de assets
LOGO_PATH      = ASSETS_DIR / "logo.png"
DETECTIVE_PNG  = ASSETS_DIR / "detetive.png"
DING_PATH      = ASSETS_DIR / "ding_chime.mp3"   # se não existir, geramos beep fallback

# Modelos

# Modelos de texto
OPENAI_MODEL_CUTS  = "gpt-4o"
OPENAI_MODEL_FACT  = "gpt-4o"
OPENAI_MODEL_INTRO = "gpt-4o"
OPENAI_MODEL_SUM   = "gpt-4o"

# Imagens (mantém)
OPENAI_IMAGE_MODEL = "gpt-image-1"


# Whisper local
WHISPER_SIZE    = "large-v3-turbo"   # tiny, base, small, medium, large-v3
WHISPER_DEVICE  = "cpu"
WHISPER_COMPUTE = "int8"

ENABLE_TITLE_EMOJI = True  # coloque False se quiser desativar rápido


# Parâmetros dos cortes
TOP_N_CORTES  = 10
SHORT_MIN_S   = 15
SHORT_MAX_S   = 35

# Overlays (escala relativa à LARGURA do vídeo; calculada em pixels no runtime)
LOGO_SCALE     = 0.08
DET_SCALE      = 0.08
OVERLAY_MARGIN = 30

# Fonte padrão (Windows). No Linux: ajuste abaixo automaticamente.
DEFAULT_FONT_WIN   = "C:/Windows/Fonts/arial.ttf"
DEFAULT_FONT_LINUX = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Vinheta
VIGNETTE_DUR = 1.0  # 1 segundo

# Veredito: fades e duração mínima
VEREDITO_DUR_MIN     = 3.0
VEREDITO_FADE_IN     = 0.50
VEREDITO_FADE_OUT_D  = 0.60

# Shorts verticais
VERTICAL_BG_MODE = "blur"   # "blur" (recomendado) ou "black"

import re
from pathlib import Path



STOP_PT = {
    "a","o","os","as","de","da","do","das","dos","e","é","em","no","na","nos","nas",
    "que","um","uma","para","por","com","como","se","ao","à","às","aos","ou","já",
    "não","mais","menos","muito","pouco","sobre","entre","até","isso","isso","esta","este",
    "ser","ter","vai","vem","foi","era","são","tem","há","dos","das","num","numa"
}


def openai_request_with_retry(request_func, *args, max_retries=3, delay=5, **kwargs):
    """
    Executa uma chamada à OpenAI com retry automático (3x) em caso de erro transitório.
    """
    import time
    for tentativa in range(1, max_retries + 1):
        try:
            return request_func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ Erro na chamada OpenAI (tentativa {tentativa}/{max_retries}): {e}")
            if tentativa < max_retries:
                print(f"⏳ Aguardando {delay}s antes de tentar novamente...")
                time.sleep(delay)
            else:
                print(f"❌ Falha após {max_retries} tentativas. Abortando requisição.")
                raise


import requests

def escolher_melhor_elevenlabs_key():
    """Seleciona a ElevenLabs key com mais créditos restantes."""
    melhor_key = None
    melhor_credito = -999999
    print("🔍 Testando chaves ElevenLabs...")

    for key in ELEVEN_API_KEYS:
        try:
            r = requests.get("https://api.elevenlabs.io/v1/user/subscription",
                             headers={"xi-api-key": key}, timeout=10)
            if r.status_code != 200:
                print(f"⚠️ Falha ao consultar key {key[:10]}... {r.status_code}")
                continue

            data = r.json()
            usado = data.get("character_count", 0)
            limite = data.get("character_limit", 0)
            restante = limite - usado

            print(f"🧾 Créditos ElevenLabs: {usado:,}/{limite:,} usados → Restantes: {restante:,}")

            if restante > melhor_credito:
                melhor_credito = restante
                melhor_key = key

        except Exception as e:
            print(f"⚠️ Erro ao verificar key {key[:10]}: {e}")

    if not melhor_key or melhor_credito <= 0:
        print("❌ Nenhuma key com créditos disponíveis! Encerrando...")
        sys.exit(1)

    print(f"✅ Usando key ElevenLabs com créditos disponíveis ({melhor_credito:,} restantes).")
    return melhor_key

def _tokset(txt: str):
    toks = re.findall(r"\b[\wáéíóúâêôãõç]+(?:-[\wáéíóúâêôãõç]+)?\b", (txt or "").lower(), flags=re.UNICODE)
    return {t for t in toks if t not in STOP_PT and len(t) > 2}


import requests

def upload_to_facebook(video_path, title, description):
    """
    Faz upload de um vídeo para uma página do Facebook.
    Preferência por vídeos verticais (_vertical.mp4).
    Garante descrição curta (<200 chars) e até 3 hashtags.
    """
    try:
        # 1️⃣ Prioriza versão vertical
        vertical_candidate = str(video_path).replace(".mp4", "_vertical.mp4")
        if os.path.exists(vertical_candidate):
            print(f"📱 Detectado vídeo vertical, usando {Path(vertical_candidate).name} para upload.")
            video_path = vertical_candidate
        else:
            print(f"ℹ️ Nenhum vídeo vertical encontrado, usando {Path(video_path).name}.")

        # 2️⃣ Limita descrição a 200 caracteres
        if len(description) > 200:
            print("✂️ Descrição muito longa, reduzindo para 200 caracteres.")
            description = description[:197].rstrip() + "..."

        # 3️⃣ Garante até 3 hashtags no final
        hashtags = re.findall(r"#\w+", description)
        if len(hashtags) > 3:
            hashtags = hashtags[:3]
        # Remove hashtags duplicadas e remonta descrição limpa
        description = re.sub(r"#\w+", "", description).strip()
        if hashtags:
            description += "\n" + " ".join(hashtags)

        # 4️⃣ Upload para Facebook Graph API
        url = f"https://graph-video.facebook.com/v18.0/{FACEBOOK_PAGE_ID}/videos"
        with open(video_path, 'rb') as video_file:
            files = {'source': video_file}
            data = {
                'title': title,
                'description': description,
                'access_token': FACEBOOK_ACCESS_TOKEN
            }

            print(f"📤 Enviando vídeo para Facebook: {Path(video_path).name} …")
            response = requests.post(url, files=files, data=data)
            result = response.json()

        if 'id' in result:
            print(f"✅ Vídeo publicado com sucesso! ID: {result['id']}")
        else:
            print(f"⚠️ Falha ao publicar vídeo: {result}")

    except Exception as e:
        print(f"❌ Erro no upload para o Facebook: {e}")


def _jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); uni = len(a | b)
    return inter / uni if uni else 0.0

def _sec(t: str) -> float:
    parts = [float(x) for x in str(t).split(":")]
    if len(parts) == 1: return parts[0]
    if len(parts) == 2: return parts[0]*60 + parts[1]
    h,m,s = parts
    if s >= 60:
        m += int(s // 60); s = s % 60
    if m >= 60:
        h += int(m // 60); m = m % 60
    return h*3600 + m*60 + s


def detectar_nome_podcast(video_titulo_original: str) -> str:
    """
    Detecta dinamicamente o nome do podcast com base no título original do vídeo.
    Exemplo: "SERGIO SACANI E BRENO MASI - Flow #434" -> "Flow Podcast"
    """
    if not video_titulo_original:
        return "podcast original"

    titulo_lower = video_titulo_original.lower()

    # Regras mais comuns (ordem importa)
    if "flow" in titulo_lower:
        return "Flow Podcast"
    if "venus" in titulo_lower:
        return "Vênus Podcast"
    if "inteligência ltda" in titulo_lower or "inteligencia ltda" in titulo_lower:
        return "Inteligência Ltda Podcast"
    if "podpah" in titulo_lower:
        return "Podpah Podcast"
    if "ciência sem fim" in titulo_lower or "ciencia sem fim" in titulo_lower:
        return "Ciência Sem Fim Podcast"
    if "cortes" in titulo_lower:
        return "Canal de Cortes"
    if "podcast" in titulo_lower:
        # genérico
        idx = titulo_lower.index("podcast")
        return video_titulo_original[max(0, idx - 20):idx + 8].strip().title()

    return "Podcast"


def escolher_emoji_para_titulo(contexto: str, tipo: str = "short") -> str:
    """
    Retorna 1 emoji coerente com o tema. Fallback seguro.
    tipo: "short" | "long"
    """
    if not contexto:
        return "⚡" if tipo == "short" else "🧠"

    ctx = contexto.lower()

    # Palavras-chave → emoji
    regras = [
        (("nasa", "espaço", "orbita", "microgravidade", "space", "spacex", "astronauta"), "🚀"),
        (("china", "tecnologia", "ia", "inteligência artificial", "chip", "snapdragon"), "🤖"),
        (("ciên", "astrono", "física", "química", "pesquisa", "ocde", "onu", "ibge"), "🧪"),
        (("economia", "mercado", "inflação", "dólar", "investimento"), "📈"),
        (("política", "governo", "eleição", "congresso", "supremo", "senado"), "🏛️"),
        (("podcast", "flow", "entrevista", "debate", "conversa"), "🎙️"),
        (("verdadeiro", "fato", "confirmado", "realidade"), "✅"),
        (("mentira", "falso", "fake", "enganado"), "❌"),
        (("surpreendente", "incrível", "descoberta"), "😲"),

    ]

    for palavras, emoji in regras:
        if any(p in ctx for p in palavras):
            return emoji

    return "⚡" if tipo == "short" else "🧠"


def transcript_text_between(segments:list, inicio_hhmmss:str, fim_hhmmss:str, pre=0.8, post=0.5) -> str:
    """Une textos de segmentos que intersectam [inicio-pre, fim+post]."""
    t0 = max(0.0, _sec(inicio_hhmmss) - pre)
    t1 = _sec(fim_hhmmss) + post
    parts = []
    for s in segments:
        # assume s tem chaves: {"start":float,"end":float,"text":str}
        ss = float(s.get("start", 0)); ee = float(s.get("end", 0))
        if ss <= t1 and ee >= t0:
            parts.append(s.get("text",""))
    return " ".join(parts).strip()

def save_debug_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")


# ============================== HELPERS BASE ===============================
def run(cmd, check=True, text=False):
    """Wrapper subprocess.run com log limpo e erro detalhado."""
    try:
        res = subprocess.run(cmd, check=check,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=text,
                             encoding="utf-8",  # 👈 Adicione esta linha
                             errors="ignore"
                             )
        return res
    except subprocess.CalledProcessError as e:
        print("❌ ERRO no comando:", " ".join(map(str, cmd)))
        try:
            sys.stderr.write(e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="ignore"))
        except Exception:
            sys.stderr.write(str(e))
        raise

def clean_dirs():
    """Apaga e recria as pastas de saída para rodar 'limpo'."""
    for p in [OUT_DIR, SHORTS_DIR, LONG_DIR]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    print("✅ Pastas limpas e recriadas.")

# ============================== FFPROBE/FFMPEG =============================
def ffprobe_resolution(video_path: Path):
    res = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path)
    ], text=True)
    w_str, h_str = res.stdout.strip().split("x")
    return int(w_str), int(h_str)

def ffprobe_duration(media_path: Path) -> float:
    res = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path)
    ], text=True)
    try:
        return float(res.stdout.strip())
    except Exception:
        return 0.0

import re, subprocess, sys

def run_ffmpeg_with_progress(cmd):
    """Executa ffmpeg exibindo progresso em uma única linha limpa."""
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
    time_pattern = re.compile(r"time=(\d+:\d+:\d+\.\d+)")
    last_time = ""
    spinner = ["⏳", "⌛", "🌀", "🔄"]
    idx = 0

    while True:
        line = process.stderr.readline()
        if not line:
            break

        match = time_pattern.search(line)
        if match:
            last_time = match.group(1)
            # Sobrescreve a linha anterior
            sys.stdout.write(f"\r{spinner[idx % len(spinner)]} Processando vídeo… tempo atual: {last_time}   ")
            sys.stdout.flush()
            idx += 1

    process.wait()
    print("\r✅ ffmpeg concluído.                              ")

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def hflip_eq_pitch_cut(src_video: Path, out_path: Path):
    import os

    # 🚫 Verificação robusta antes de processar
    if not os.path.exists(src_video):
        print(f"⚠️ Arquivo não encontrado: {src_video}, pulando correção.")
        return

    file_size = os.path.getsize(src_video)
    if file_size < 2 * 1024 * 1024:  # menor que 2 MB
        print(f"⚠️ Arquivo muito pequeno ({file_size / 1024:.1f} KB): {src_video}, pulando correção.")
        return

    print("▶ Aplicando correção sutil de imagem…")
    vf = "eq=saturation=1.06:contrast=1.06:gamma=1.02"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_video),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path)
    ]
    run_ffmpeg_with_progress(cmd)

def extract_wav_mono16k(src_video: Path, wav_out: Path):
    print("▶ Extraindo WAV mono 16 kHz…")
    cmd = ["ffmpeg", "-y", "-i", str(src_video), "-ac", "1", "-ar", "16000", "-vn", str(wav_out)]
    run(cmd)
    print(f"✅ WAV: {wav_out}")

def fade_out_audio(video_in: Path, out_path: Path, fade_s: float = 0.35):
    dur = ffprobe_duration(video_in)
    if dur <= 0 or fade_s <= 0:
        shutil.copyfile(video_in, out_path)
        return
    start = max(0.0, dur - fade_s)
    cmd = [
        "ffmpeg","-y","-i", str(video_in),
        "-af", f"afade=t=out:st={start:.2f}:d={fade_s:.2f}",
        "-c:v","copy","-c:a","aac","-b:a","192k",
        "-movflags","+faststart", str(out_path)
    ]
    run(cmd)

# ============================== YT-DLP DOWNLOAD ============================
import subprocess, sys
from pathlib import Path

def run_with_live_output(cmd_list):
    """Executa subprocesso exibindo progresso em uma única linha limpa."""
    process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_percent = ""
    for line in process.stdout:
        if "%" in line:
            # tenta capturar algo como " 23.4%" do yt-dlp
            match = re.search(r"(\d{1,3}\.\d)%", line)
            if match:
                percent = match.group(1)
                if percent != last_percent:
                    sys.stdout.write(f"\r📥 Baixando vídeo... {percent}%   ")
                    sys.stdout.flush()
                    last_percent = percent
        elif "Destination" in line:
            # exibe o nome do arquivo só uma vez
            print("\n" + line.strip())
        elif "Deleting original file" in line:
            continue  # ignora linhas de limpeza
    process.wait()
    print("\r✅ Download concluído.           ")
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd_list)
    return process.returncode



def download_youtube_video(url, out_dir):
    print("ℹ️  Baixando vídeo…")
    print("ℹ️  Usando cookies.txt (se disponível)…")

    cookies_file = os.path.join(Path(".").resolve(), "cookies.txt")

    base_cmd = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio/best",
        "-o", os.path.join(out_dir, "%(title)s.%(ext)s"),
        "--merge-output-format", "mp4",
        "--progress", "--console-title"
    ]

    def get_last_downloaded_mp4():
        """Retorna o último arquivo MP4 baixado na pasta de saída."""
        mp4s = list(Path(out_dir).glob("*.mp4"))
        if not mp4s:
            return None
        return max(mp4s, key=lambda f: f.stat().st_mtime)

    # 1️⃣ tenta sem cookies (funciona na maioria dos casos)
    try:
        print("▶ Tentando sem cookies (bestvideo+bestaudio)…")
        run_with_live_output(base_cmd + [url])
        video_path = get_last_downloaded_mp4()
        if video_path:
            print(f"✅ Download concluído sem cookies: {video_path.name}")
            return video_path
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Falha sem cookies ({e.returncode}) — tentando cookies.txt…")

    # 2️⃣ tenta com cookies.txt
    if os.path.exists(cookies_file):
        try:
            print("▶ Tentando com cookies.txt…")
            run_with_live_output(base_cmd + ["--cookies", cookies_file, url])
            video_path = get_last_downloaded_mp4()
            if video_path:
                print(f"✅ Download concluído com cookies.txt: {video_path.name}")
                return video_path
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Falha com cookies.txt ({e.returncode}) — tentando cookies do Chrome…")
    else:
        print("⚠️  cookies.txt não encontrado — tentando usar cookies do Chrome…")

    # 3️⃣ tenta com cookies do Chrome
    try:
        print("▶ Tentando com cookies do Chrome…")
        run_with_live_output(base_cmd + ["--cookies-from-browser", "chrome", url])
        video_path = get_last_downloaded_mp4()
        if video_path:
            print(f"✅ Download autenticado via Chrome concluído: {video_path.name}")
            return video_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Falha ao baixar o vídeo mesmo com cookies do Chrome (código {e.returncode}).")

    print("❌ Nenhuma tentativa de download foi bem-sucedida.")
    sys.exit(1)


# ============================== TRANSCRIÇÃO/DIARIZAÇÃO =====================
def transcribe_whisper(wav_path: Path, model_size=WHISPER_SIZE):
    """Transcreve áudio com Whisper local exibindo progresso em uma única linha (otimizada)."""
    import os, json, time, sys
    from faster_whisper import WhisperModel

    os.environ["CT2_THREAD_COUNT"] = "16"  # usa 8 threads (ajuste conforme CPU)

    cache_path = wav_path.with_suffix(".transcricao_cache.json")
    if cache_path.exists():
        print(f"⚡ Usando transcrição em cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("▶ Transcrevendo (Whisper local)…")
    start_time = time.time()
    model = WhisperModel(
    model_size,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE,
    cpu_threads=os.cpu_count(),
    num_workers=4
)

    segments, info = model.transcribe(
        str(wav_path),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=150),  # pausas curtas = +fluidez
        beam_size=5,
        word_timestamps=False
    )

    out = []
    total_duration = info.duration if hasattr(info, "duration") else None
    processed = 0.0
    last_update = 0
    spinner = ["🧠", "💭", "⏳", "🔄"]
    idx = 0

    for s in segments:
        out.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
        processed = float(s.end)
        if total_duration:
            progress = min(100, (processed / total_duration) * 100)
            if int(progress) != last_update:
                sys.stdout.write(
                    f"\r{spinner[idx % len(spinner)]} Transcrevendo... {progress:5.1f}% | tempo: {processed:6.1f}s"
                )
                sys.stdout.flush()
                idx += 1
                last_update = int(progress)

    elapsed = time.time() - start_time
    print(f"\r✅ Transcrição concluída: {len(out)} segmentos extraídos em {elapsed:.1f}s.{' '*20}")

    # Salva cache local (para reuso em runs futuros)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out

def diarize_local(audio_path: Path):
    print("▶ Diarizando (Resemblyzer + SpectralClusterer)…")
    import librosa
    wav, sr = librosa.load(audio_path, sr=16000)
    enc = VoiceEncoder("cpu")
    step = sr
    frames = [wav[i:i+step] for i in range(0, len(wav), step)]
    embeds = np.array([enc.embed_utterance(f) for f in frames if len(f) > 1000])
    if embeds.size == 0:
        dur = len(wav)/sr
        print("⚠️  Sem frames suficientes. Assumindo 1 falante.")
        return [{"start": 0.0, "end": dur, "speaker": "SPEAKER_00"}]
    if embeds.ndim == 1:
        embeds = embeds.reshape(-1, 1)
    try:
        labels = SpectralClusterer(min_clusters=1, max_clusters=4).predict(embeds)
    except Exception as e:
        print(f"⚠️  Falha na clusterização: {e}. Usando 1 falante único.")
        labels = np.zeros(len(embeds))
    if len(np.unique(labels)) <= 1:
        dur = len(wav) / sr
        diarization = [{"start": 0.0, "end": dur, "speaker": "SPEAKER_00"}]
    else:
        diarization = []
        last = labels[0]; seg_start = 0.0
        for i in range(1, len(labels)):
            if labels[i] != last:
                diarization.append({
                    "start": float(i-1), "end": float(i),
                    "speaker": f"SPEAKER_{int(last):02d}"
                })
                seg_start = i; last = labels[i]
        diarization.append({
            "start": float(seg_start), "end": float(len(labels)),
            "speaker": f"SPEAKER_{int(last):02d}"
        })
    print(f"✅ Falantes detectados: {len(np.unique(labels))}")
    return diarization

def assign_speakers(asr_segments, spk_segments):
    def overlap(a0,a1,b0,b1): return max(0.0, min(a1,b1) - max(a0,b0))
    assigned = []
    for s in asr_segments:
        scores = {}
        for d in spk_segments:
            ov = overlap(s["start"], s["end"], d["start"], d["end"])
            if ov > 0:
                scores[d["speaker"]] = scores.get(d["speaker"], 0.0) + ov
        spk = max(scores, key=scores.get) if scores else (assigned[-1]["speaker"] if assigned else "Pessoa 1")
        assigned.append({**s, "speaker": spk})
    return assigned

def write_transcript(segments, out_path: Path):
    def human_ts(sec):
        if sec is None: return "00:00:00.000"
        sec = max(0.0, float(sec))
        total_ms = int(round(sec*1000))
        h = (total_ms // 3600000) % 100
        m = (total_ms // 60000) % 60
        s = (total_ms // 1000) % 60
        ms= total_ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{human_ts(s['start'])} – {human_ts(s['end'])}] {s['speaker']}: {s['text']}\n")
    print(f"✅ Transcrição salva: {out_path}")

def build_context_from_segments(segments):
    return "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)

# ============================== OPENAI (CORTES/FACT/INTRO/SUM) =============
def openai_client():
    return OpenAIChat(api_key=OPENAI_API_KEY)

def openai_images_client():
    return OpenAIImages(api_key=OPENAI_API_KEY)

def chunk_text(txt, max_chars=8000):
    blocks, cur = [], []
    total = 0
    for line in txt.splitlines():
        if total + len(line) + 1 > max_chars and cur:
            blocks.append("\n".join(cur)); cur, total = [], 0
        cur.append(line); total += len(line) + 1
    if cur: blocks.append("\n".join(cur))
    return blocks

import time
import json
from datetime import datetime

def gpt_sugerir_cortes(segments: list):
    """
    Gera cortes automáticos com base em timestamps reais e contexto expandido.
    O GPT agora considera as falas anteriores e posteriores para evitar cortes abruptos.
    """
    client = openai_client()

    SHORT_MIN_S = 8
    SHORT_MAX_S = 40
    DEBUG_CORTES = True
    DEBUG_LOG_FILE = OUT_DIR / "debug_gpt_cortes_log.jsonl"

    def append_debug_log(entry: dict):
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def chunk_segments(segments, max_chars=8000):
        blocks, current, total = [], [], 0
        for s in segments:
            js = json.dumps(s, ensure_ascii=False)
            if total + len(js) > max_chars and current:
                blocks.append(current)
                current, total = [], 0
            current.append(s)
            total += len(js)
        if current:
            blocks.append(current)
        return blocks

    # 👉 Novo: incluir ±10s de contexto extra no prompt
    def context_preview(segments):
        return "\n".join(
            f"[{s['start']:.1f}–{s['end']:.1f}] {s['speaker']}: {s['text']}"
            for s in segments[:20]
        )

    blocks = chunk_segments(segments)
    all_cortes = []
    print("▶ Pedindo cortes ao GPT com base em timestamps reais e contexto expandido…")

    for i, blk in enumerate(blocks, 1):
        print(f"   - Bloco {i}/{len(blocks)} com {len(blk)} falas")

        prompt = f"""
Você é um editor profissional de vídeos curtos para YouTube Shorts e TikTok.

Abaixo estão falas transcritas de um vídeo, com timestamps de início e fim (em segundos).

Sua tarefa:
- Escolher de 3 a 6 trechos impactantes, curiosos, polêmicos ou emocionais;
- Cada corte deve começar até 3 segundos antes da fala principal e terminar 1 segundo depois;
- A duração deve ficar entre {SHORT_MIN_S} e {SHORT_MAX_S} segundos;
- Prefira cortes entre 20 e 40 segundos, com frases de impacto ou momentos de forte emoção.
- Foque em momentos com tensão, emoção, humor ou informação que gera engajamento;
- **Considere também as falas anteriores e posteriores** para manter o contexto e evitar cortes secos;
- Ignore falas monótonas ou sem emoção.
- Ignore totalmente trechos de propaganda, merchandising ou patrocínio.
- Descarte qualquer trecho que mencione:
  "patrocínio", "oferecimento", "parceria", "cupom", "link na descrição",
  "use o cupom", "insider", "blaze", "sponsor", "anunciante", "marca", "produto", "lojinha", "merch", "parceiro", "camisa", "camiseta", "shop", "app", "inscreva-se", "baixe o app".


Responda **apenas em JSON válido** (lista de objetos):
[
  {{
    "inicio": "HH:MM:SS",
    "fim": "HH:MM:SS",
    "titulo": "curto e chamativo",
    "descricao": "resumo da fala",
    "hashtags": ["#exemplo1", "#exemplo2"]
  }}
]

Trechos do vídeo (com contexto expandido ±10s):
{context_preview(blk)}

Falas (JSON detalhado):
{json.dumps(blk, ensure_ascii=False, indent=2)}
"""

        start_time = time.time()
        try:
            resp = openai_request_with_retry(
                client.chat.completions.create,
                model=OPENAI_MODEL_CUTS,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )

            try:
                usage = resp.usage
                openai_stats["requests"] += 1
                openai_stats["prompt_tokens"] += usage.prompt_tokens
                openai_stats["completion_tokens"] += usage.completion_tokens
                openai_stats["total_tokens"] += usage.total_tokens
            except Exception:
                pass

            elapsed = time.time() - start_time
            text = resp.choices[0].message.content.strip()

            debug_entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "block_index": i,
                "falas_enviadas": len(blk),
                "response_chars": len(text),
                "elapsed_seconds": round(elapsed, 2),
                "model": OPENAI_MODEL_CUTS,
            }
            append_debug_log(debug_entry)

            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[len("json"):].strip()

            data = json.loads(text)
            for item in data:
                ini = hhmmss_to_s(item["inicio"])
                fim = hhmmss_to_s(item["fim"])
                dur = fim - ini
                if dur > SHORT_MAX_S:
                    fim = ini + SHORT_MAX_S
                elif dur < SHORT_MIN_S:
                    fim = ini + SHORT_MIN_S

                def fmt(sec):
                    h = int(sec // 3600)
                    m = int((sec % 3600) // 60)
                    s = int(sec % 60)
                    return f"{h:02d}:{m:02d}:{s:02d}"

                item["inicio"] = fmt(ini)
                item["fim"] = fmt(fim)
                all_cortes.append(item)

            raw_path = OUT_DIR / f"gpt_cortes_raw_bloco{i}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"💾 Resposta bruta salva: {raw_path}")

        except Exception as e:
            print(f"⚠️  Erro ao processar bloco {i}: {e}")
            append_debug_log({"block_index": i, "error": str(e)})
            continue

    print(f"✅ Total de cortes sugeridos: {len(all_cortes)}")
    print(f"📜 Log detalhado salvo em: {DEBUG_LOG_FILE}")



    return all_cortes


def hhmmss_to_s(hms: str) -> float:
    parts = [int(x) for x in hms.split(":")]
    return parts[0]*3600 + parts[1]*60 + parts[2]

def filtra_e_otimiza_topN(cortes, topN=TOP_N_CORTES, min_s=8, max_s=55):
    """
    Filtra cortes com base em duração e remove duplicados ou tempos inválidos.
    Corrige tempos fora do formato e descarta cortes invertidos.
    """
    def parse_time_safe(t):
        try:
            parts = str(t).split(":")
            parts = [float(x) for x in parts]
            if len(parts) == 1:
                return parts[0]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
            elif len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
        except Exception:
            return 0.0
        return 0.0

    valid = []
    for c in cortes:
        ini = parse_time_safe(c.get("inicio", "0"))
        fim = parse_time_safe(c.get("fim", "0"))
        dur = max(0, fim - ini)

        if dur <= 0:
            print(f"⚠️ Corte descartado: tempo invertido ou nulo ({ini:.2f}s → {fim:.2f}s)")
            continue

        if not (min_s <= dur <= max_s):
            print(f"⏱️ Corte fora da faixa ({dur:.1f}s): {c.get('titulo','')}")
            continue

        c["duracao"] = dur
        valid.append(c)

    if not valid:
        print("⚠️ Nenhum corte válido após filtragem.")
        return []

    # Ordena por duração (desc) — ou pode ser aleatório, se preferir variedade
    valid.sort(key=lambda x: x["duracao"], reverse=True)

    # Evita cortes sobrepostos (mesmos tempos)
    uniq = []
    vistos = set()
    for c in valid:
        key = (round(parse_time_safe(c["inicio"]), 1), round(parse_time_safe(c["fim"]), 1))
        if key not in vistos:
            vistos.add(key)
            uniq.append(c)

    top = uniq[:topN]
    print(f"✅ Top-N pronto ({len(top)}).")
    return top

import requests

import requests
import sys


def gerar_abertura_por_veredito(veredito):
    veredito = veredito.lower()
    if "verdadeiro" in veredito or "confirmado" in veredito:
        return random.choice([
            "A verdade é que",
            "Sim, isso realmente aconteceu:",
            "Fato confirmado:"
        ])
    elif "impreciso" in veredito:
        return random.choice([
            "Essa afirmação gera dúvida:",
            "Nem tudo aqui está confirmado:",
            "Os dados ainda são inconclusivos:"
        ])
    elif "relato pessoal" in veredito:
        return random.choice([
            "Aqui vai uma visão pessoal:",
            "Esse trecho mostra uma experiência vivida, não um dado verificável:",
            "Nesse ponto, o convidado compartilha uma percepção própria:"
        ])
    elif "falso" in veredito or "enganoso" in veredito:
        return random.choice([
            "Essa não é bem assim:",
            "Os fatos desmentem essa ideia:",
            "O que foi dito não bate com os dados:"
        ])
    else:
        return random.choice([
            "Nesse trecho, o papo é mais descontraído:",
            "Aqui o assunto foge um pouco dos fatos:",
            "Esse momento é mais sobre opinião do que informação."
        ])


import time
from datetime import datetime

def fact_check_cortes(cortes, segments):
    """
    Checagem de fatos aprimorada:
    - contexto ampliado (±10s)
    - nova categoria 'Relato pessoal'
    - filtro de trechos irrelevantes
    """
    client = openai_client()
    DEBUG_LOG_FILE = OUT_DIR / "debug_factcheck_log.jsonl"

    def append_debug(entry):
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    resultados = []
    print("▶ Checando veracidade (modo alinhado e contextualizado)…")

    # expressões típicas de trechos irrelevantes
    irrelevantes = [
        "valeu galera", "curte o vídeo", "curtam o vídeo", "deixa o like", "se inscreve",
        "encerrando", "até a próxima", "obrigado por assistir", "tamo junto",
        "final do episódio", "segue a gente", "ativa o sininho"
    ]

    for i, c in enumerate(cortes, 1):
        inicio = c.get("inicio", "00:00:00")
        fim = c.get("fim", "00:00:00")

        # 🧠 contexto ampliado (±10s)
        texto_real = transcript_text_between(segments, inicio, fim, pre=10.0, post=10.0)
        texto_real_lower = texto_real.lower()

        # 🚫 filtro de trechos irrelevantes
        if any(p in texto_real_lower for p in irrelevantes):
            print(f"⚠️ Corte {i} ignorado (trecho irrelevante).")
            continue

        save_debug_text(OUT_DIR / f"debug_factcheck/slice_short_{i:02d}.txt", texto_real)

        # 🧩 prompt com nova categoria
        prompt = f"""
        Você é um jornalista especializado em verificação de fatos de podcasts e vídeos curtos.

        Analise o trecho abaixo (com ±10 segundos de contexto).  
        Identifique se há uma afirmação factual e classifique em uma das categorias:

        - **Verdadeiro** → confirmado por fontes oficiais reconhecidas.
        - **Falso** → contradiz dados ou estudos confiáveis.
        - **Impreciso** → menciona fatos parcialmente corretos ou sem dados públicos suficientes.
        - **Relato pessoal** → opinião, emoção ou experiência individual.
        - **Irrelevante** → agradecimentos, autopromoção, pedidos de like, etc.

        Trecho analisado:
        \"\"\"{texto_real.strip()}\"\"\"


        Se houver fato verificável, cite uma fonte confiável (ex: IBGE, ONU, OMS, IFBB, etc.).  

        Agora gere também uma narração curta (2–3 frases) adequada para um vídeo curto do YouTube, seguindo estas regras:
        - Use um tom informativo, natural e curioso — como um apresentador explicando algo interessante.
        - A narração deve soar envolvente e com leve curiosidade, sem parecer um telejornal nem um comercial.
        - Evite redundâncias e mantenha frases curtas e diretas (máximo 20 palavras cada).
        - Comece com um gancho leve e contextual.
        - Explique o ponto principal de forma clara.
        - Termine com a conclusão do veredito (verdadeiro, falso, impreciso ou relato pessoal).
        - **Não inclua convites, chamadas ou teasers** (ex: nada de “fique conosco”, “veja até o fim”).
        - A narração deve ser autossuficiente, sem depender de outros trechos do podcast.
        - **Não repita o veredito** se ele já estiver implícito no texto.
        - Use frases curtas (máx. 20 palavras) e mantenha a narração entre 2 e 3 frases no total.


        Responda em JSON válido no formato:
        {{
          "veredito": "Verdadeiro/Falso/Impreciso/Relato pessoal/Irrelevante",
          "fonte": "Nome da fonte ou N/A",
          "narracao_youtube": "Texto narrado em tom leve, natural e educativo"
        }}
        """

        try:
            resp = openai_request_with_retry(
                client.chat.completions.create,
                model=OPENAI_MODEL_FACT,
                temperature=0.6,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )

            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip("`\n ")

            data = json.loads(text)

            veredito = data.get("veredito", "Impreciso")
            fonte = data.get("fonte", "N/A")
            narr_y = data.get("narracao_youtube", "").strip()

            # evita processar curtas irrelevantes
            if veredito.lower() == "irrelevante":
                print(f"⚠️ Corte {i} classificado como irrelevante, pulando.")
                continue

            # 🎙️ hooks e CTA
            # 🎙️ Escolhe abertura coerente com o veredito
            abertura = gerar_abertura_por_veredito(veredito)
            narr_y = f"{abertura} {narr_y}"
            if not narr_y.endswith(('.', '!', '…')):
                narr_y += '.'

            # 🎯 CTA (variações naturais para o encerramento)
            ctas = [
                "Curta e inscreva-se para mais verificações como esta!",
                "Acompanhe o canal e fique por dentro de mais checagens de fatos!",
                "Gostou do vídeo? Deixe o like e compartilhe com seus amigos!",
                "Inscreva-se para descobrir mais verdades e mitos em podcasts!"
            ]
            narr_y += " " + random.choice(ctas)

            resultados.append({
                **c,
                "status": veredito,
                "fonte": fonte,
                "narracao_youtube": narr_y
            })

            print(f"\n🧠 Fact-check {i}: {c.get('titulo','(sem título)')}")
            print(f"→ {veredito} — {fonte}")
            print(f"🎙️ Narr_YT: {narr_y[:180]}{'...' if len(narr_y)>180 else ''}")

            append_debug({"index": i, "titulo": c.get("titulo"), "veredito": veredito, "fonte": fonte})

        except Exception as e:
            print(f"❌ Erro no fact-check {i}: {e}")
            append_debug({"index": i, "error": str(e)})

    fc_json = OUT_DIR / "shorts_top10_verificados.json"
    with open(fc_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON consolidado salvo em: {fc_json}")

    return resultados



def gpt_intro_texto(tres_cortes):
    """Gera texto de introdução curto e dinâmico com base nos 3 cortes."""
    client = openai_client()
    resumo = "\n".join([f"- {c['titulo']} (veredito: {c.get('checagem',{}).get('status','?')})" for c in tres_cortes])
    prompt = f"""
Gere uma INTRO de 2-3 frases, tom leve e confiante, para abrir um vídeo de fact-check.
Mostre que vamos checar rapidamente 3 pontos e revelar os vereditos. Não cite URLs.

Baseie-se nestes tópicos:
{resumo}

Responda com um parágrafo curto e natural em PT-BR.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL_INTRO, temperature=0.6,
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip()

def gerar_metadados_facebook(title_y, desc_y, tags_y, client):
    try:
        prompt = f"""
Otimize o título, descrição e hashtags para Facebook Reels
com base nos metadados do YouTube abaixo:

Título: {title_y}
Descrição: {desc_y}
Tags: {', '.join(tags_y)}

Reescreva para o público do Facebook Reels:
- Título: curto, direto, até 80 caracteres.
- Se houver uma pessoa notória (convidado famoso, personalidade ou nome reconhecido), o título deve COMEÇAR com esse nome.
  Exemplo: "🔥 Ramon Dino comenta genética no Flow Podcast"
- Descrição: envolvente, até 200 caracteres, convite à interação.
- Hashtags: 3–6 curtas e relevantes, sem acento.
- O texto deve refletir que o vídeo é curto e direto (estilo Reels/Shorts).

Responda APENAS em JSON válido:
{{
  "title": "Título para Facebook",
  "description": "Descrição curta e envolvente",
  "hashtags": ["#exemplo", "#factcheck", "#podcast"]
}}
"""
        resp = openai_request_with_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip("`\n ")

        data = json.loads(raw)
        title_fb = data.get("title", title_y)
        desc_fb = data.get("description", desc_y)
        hashtags_fb = data.get("hashtags", [])
        return title_fb, desc_fb, hashtags_fb

    except Exception as e:
        print(f"⚠️ Erro ao gerar metadados para Facebook: {e}")
        return title_y, desc_y, tags_y


def resumir_conteudo_video(transcricao_txt: str) -> str:
    """Resumo curto (estilo manchete) para embasar a thumbnail."""
    try:
        client = openai_client()
        prompt = (
            "Resuma em uma única frase curta e chamativa o tema principal do vídeo abaixo. "
            "Estilo: manchete de YouTube — direto, curioso e informativo.\n\n"
            f"Transcrição:\n{transcricao_txt[:5000]}"
        )
        resp = openai_request_with_retry(
            client.chat.completions.create,
            model=OPENAI_MODEL_SUM,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            usage = resp.usage
            openai_stats["requests"] += 1
            openai_stats["prompt_tokens"] += usage.prompt_tokens
            openai_stats["completion_tokens"] += usage.completion_tokens
            openai_stats["total_tokens"] += usage.total_tokens
        except Exception:
            pass

        resumo = resp.choices[0].message.content.strip()
        print(f"🧠 Resumo gerado: {resumo}")
        return resumo
    except Exception as e:
        print(f"⚠️ Erro ao resumir conteúdo: {e}")
        return "Debate sobre fatos e verdades."

# ============================== CUT/OVERLAYS/VINHETA =======================
def cut_short(src_video: Path, ini_hms: str, fim_hms: str, out_path: Path):
    """
    Corta o vídeo com margens maiores (+8s início, +4s fim) e checa duração real
    para evitar cortes inválidos ("sem vídeo").
    """
    ini = hhmmss_to_s(ini_hms)
    fim = hhmmss_to_s(fim_hms)

    # Segurança contra valores fora dos limites do vídeo
    dur_total = ffprobe_duration(src_video)
    if dur_total <= 0:
        print(f"⚠️ Vídeo base sem duração detectada. Pulando corte {out_path.name}.")
        return None

    ini_safe = max(0, ini - 8.0)
    fim_safe = min(dur_total - 0.2, fim + 4.0)
    if ini_safe >= fim_safe:
        print(f"⚠️ Corte inválido ({ini_safe:.2f}–{fim_safe:.2f}). Pulando.")
        return None

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{ini_safe:.2f}", "-to", f"{fim_safe:.2f}",
        "-i", str(src_video),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path)
    ]
    print(f"✂️ Cortando vídeo de {ini_safe:.2f}s até {fim_safe:.2f}s → {out_path.name}")
    run(cmd)

    # Verifica se o corte gerou arquivo válido (>1s)
    dur = ffprobe_duration(out_path)
    if dur < 1:
        print(f"⚠️ Corte gerado inválido ({dur:.2f}s). Removendo {out_path.name}.")
        out_path.unlink(missing_ok=True)
        return None

    return out_path

def overlay_logo_and_detective(video_in: Path, logo_path: Path, detective_path: Path,
                               video_out: Path, base_width: int):
    logo_w = max(1, int(base_width * LOGO_SCALE))
    det_w  = max(1, int(base_width * DET_SCALE))
    filter_complex = (
        f"[1:v]scale={logo_w}:-1[logo];"
        f"[2:v]scale={det_w}:-1[det];"
        f"[0:v][logo]overlay=x=main_w-overlay_w-{OVERLAY_MARGIN}:"
        f"y=main_h-overlay_h-{OVERLAY_MARGIN}[vl];"
        f"[vl][det]overlay=x=main_w-overlay_w-{OVERLAY_MARGIN*5}:"
        f"y=main_h-overlay_h-{OVERLAY_MARGIN*5}[vout]"
    )
    cmd = [
        "ffmpeg","-y",
        "-i", str(video_in),
        "-i", str(logo_path),
        "-i", str(detective_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(video_out)
    ]
    print(f"▶ Aplicando overlays em {video_in} …"); run(cmd); print(f"✅ Overlay aplicado: {video_out}")

def ensure_ding():
    if DING_PATH.exists(): return
    print("ℹ️  Gerando ding_chime.mp3 (fallback)…")
    cmd = ["ffmpeg","-y","-f","lavfi","-i","sine=frequency=700:duration=0.9","-c:a","libmp3lame","-q:a","5", str(DING_PATH)]
    run(cmd)

def get_system_font():
    return DEFAULT_FONT_WIN if os.name == "nt" else DEFAULT_FONT_LINUX

def make_vignette(output_path: Path, width: int, height: int):
    ensure_ding()
    text = "Cortes Verificados"
    is_windows = (os.name == "nt")
    font_path = get_system_font().replace("\\", "/")
    if is_windows:
        draw = (f"[v]drawtext=text='{text}':font='Arial':fontsize=56:"
                f"fontcolor=white:borderw=2:bordercolor=black@0.6:"
                f"x=(w-text_w)/2:y=(h-text_h)/2[vout]")
    else:
        draw = (f"[v]drawtext=text='{text}':fontsize=56:fontfile='{font_path}':"
                f"fontcolor=white:borderw=2:bordercolor=black@0.6:"
                f"x=(w-text_w)/2:y=(h-text_h)/2[vout]")
    filter_complex = (
        f"[0:v]scale={width}:{height},"
        f"fade=t=in:st=0:d=0.25,fade=t=out:st=0.75:d=0.25,"
        f"boxblur=luma_radius=6:luma_power=1,format=yuv420p,setsar=1[v];" + draw
    )
    cmd = [
        "ffmpeg","-y",
        "-loop","1","-t", f"{VIGNETTE_DUR:.2f}",
        "-i", str(LOGO_PATH),
        "-i", str(DING_PATH),
        "-filter_complex", filter_complex,
        "-map","[vout]","-map","1:a",
        "-c:v","libx264","-preset","ultrafast","-crf","20",
        "-c:a","aac","-b:a","192k","-movflags","+faststart",
        str(output_path)
    ]
    print("▶ Gerando vinheta curta (logo + som)…"); run(cmd); print(f"✅ Vinheta gerada: {output_path}")

# ============================== VEREDITO (CARD) ============================
def make_last_frame(video_path: Path, out_png: Path):
    """
    Extrai o último frame do vídeo, com fallbacks robustos (sem usar n_forced).
    """
    # 1) Tentativa: sseof -0.001
    try:
        cmd = [
            "ffmpeg","-y","-sseof","-0.001",
            "-i", str(video_path),
            "-update","1","-frames:v","1",
            str(out_png)
        ]
        run(cmd)
        if out_png.exists() and out_png.stat().st_size > 500:
            print(f"✅ Último frame salvo: {out_png}")
            return
    except Exception as e:
        print(f"⚠️ Método 1 falhou: {e}")

    # 2) Fallback: seek para (dur-0.5)
    try:
        dur = ffprobe_duration(video_path)
        seek = max(0.0, dur - 0.5)
        cmd = [
            "ffmpeg","-y","-ss", f"{seek:.2f}",
            "-i", str(video_path),
            "-frames:v","1",
            str(out_png)
        ]
        run(cmd)
        if out_png.exists() and out_png.stat().st_size > 500:
            print(f"✅ Último frame (fallback) salvo: {out_png}")
            return
    except Exception as e:
        print(f"⚠️ Método 2 falhou: {e}")

    raise RuntimeError(f"❌ Falha crítica: não foi possível gerar o último frame para {video_path}")

def make_verdict_card(
    bg_image: Path,
    detective_png: Path,
    narration_audio: Path,
    verdict: str,
    fonte: str,
    w: int,
    h: int,
    out_video: Path
):
    audio_dur = max(VEREDITO_DUR_MIN, ffprobe_duration(narration_audio))
    color_map = {
        "VERDADEIRO": "green@0.18",
        "FALSO": "red@0.25",
        "IMPRECISO": "yellow@0.25",
        "SEM DADOS SUFICIENTES": "gray@0.25",
        "SEM DADOS": "gray@0.25",
        "": "gray@0.25"
    }
    bg_color = color_map.get(verdict.upper(), "gray@0.25")
    is_windows = (os.name == "nt")
    font_path = get_system_font().replace("\\", "/")
    font_opt = "font='Arial'" if is_windows else f"fontfile='{font_path}'"

    def _esc(s: str) -> str:
        return (s.replace("\\", "\\\\").replace("'", r"\'")
                .replace(":", r"\:").replace(",", r"\,").replace("=", r"\="))

    # Nome curto da fonte: "Site X"
    fonte_label = "N/A"
    if fonte:
        m = re.search(r"https?://(?:www\d?\.)?([^/]+)", fonte)
        if m:
            dominio = m.group(1).split(".")[0].capitalize()
            fonte_label = f"Site {dominio}"
        else:
            fonte_label = fonte

    title = "Verificação de Fatos" if verdict else "Introdução"
    txt1 = _esc(title)
    txt2 = _esc(f"VEREDITO → {verdict.upper()}") if verdict else _esc("Vamos conferir os fatos")
    fonte_label = (fonte_label or "")[:120]
    txt3 = _esc(f"Fonte → {fonte_label}") if verdict else _esc("")

    # Monta o filter graph
    filter_complex = (
        f"[0:v]scale={w}:{h},boxblur=luma_radius=8:luma_power=1[bg];"
        f"[1:v]scale=153:-1[det];"
        f"[bg]drawbox=x=(iw-0.86*iw)/2:y=(ih-0.36*ih)/2:"
        f"w=0.86*iw:h=0.36*ih:color=black@0.55:t=fill[v1];"
        f"[v1]drawbox=x=0:y=0:w=iw:h=ih:color={bg_color}:t=fill[v2];"
        f"[v2]drawtext=text='{txt1}':{font_opt}:fontsize=48:"
        f"fontcolor=white:borderw=2:bordercolor=black@0.6:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-100[v3];"
        f"[v3]drawtext=text='{txt2}':{font_opt}:fontsize=64:"
        f"fontcolor=white:borderw=2:bordercolor=black@0.6:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-20[v4];"
        f"[v4]drawtext=text='{txt3}':{font_opt}:fontsize=30:"
        f"fontcolor=white:borderw=2:bordercolor=black@0.6:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+60[v5];"
        f"[v5][det]overlay=x=main_w-overlay_w-30:y=main_h-overlay_h-30[v6];"
        f"[v6]fade=t=in:st=0:d={VEREDITO_FADE_IN:.2f},"
        f"fade=t=out:st={max(0.0, audio_dur-VEREDITO_FADE_OUT_D):.2f}:"
        f"d={VEREDITO_FADE_OUT_D:.2f}[vout]"
    )

    cmd = [
        "ffmpeg","-y",
        "-loop","1","-t", f"{audio_dur:.2f}", "-i", str(bg_image),
        "-i", str(detective_png),
        "-i", str(narration_audio),
        "-filter_complex", filter_complex,
        "-map","[vout]","-map","2:a",
        "-c:v","libx264","-preset","ultrafast","-crf","20",
        "-c:a","aac","-b:a","192k","-ar","48000",
        "-shortest","-movflags","+faststart",
        str(out_video)
    ]
    print(f"▶ Gerando veredito visual: {verdict if verdict else 'INTRO'}…"); run(cmd); print(f"✅ Card salvo: {out_video}")

# ============================== SHORTS VERTICAL ============================
def make_vertical_from_16x9(input_mp4: Path, output_mp4: Path):
    """
    Converte 16:9 -> 9:16 sem cortar conteúdo:
    - Fundo borrado 1080x1920
    - Vídeo original escalado para caber (pillarbox), centralizado
    """
    if VERTICAL_BG_MODE == "blur":
        vf = (
            "split[main][bg];"
            "[main]scale=1080:-2:force_original_aspect_ratio=decrease[p1];"
            "[bg]scale=1080:1920,boxblur=luma_radius=30:luma_power=1:chroma_radius=30:chroma_power=1[p2];"
            "[p2][p1]overlay=(W-w)/2:(H-h)/2,format=yuv420p,setsar=1"
        )
    else:
        vf = (
            "scale=1080:-2:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1"
        )
    cmd = [
        "ffmpeg","-y",
        "-i", str(input_mp4),
        "-vf", vf,
        "-c:v","libx264","-preset","ultrafast","-crf","20",
        "-c:a","aac","-b:a","160k","-ar","48000",
        "-movflags","+faststart",
        str(output_mp4)
    ]
    print(f"▶ Convertendo para vertical 9:16: {output_mp4.name}…"); run(cmd); print("✅ Vertical pronto.")

# ============================== INTRO (LONG VIDEO) =========================
def tts_elevenlabs(text, out_path, speed=1.05, voice_id="nPczCjzI2devNBz1zQrb"):
    """
    Gera narração natural e masculina (Brian – estilo William Bonner / telejornal).
    Otimizada para clareza, ritmo e naturalidade em vídeos factuais.
    Com retry automático (3x) em caso de falha na API.
    """
    import re, requests, time, subprocess, os

    # === 1️⃣ Tratamento do texto para pausas e fluidez natural ===
    text = text.strip()

    global eleven_stats
    char_count = len(text)
    eleven_stats["requests"] += 1
    eleven_stats["characters"] += char_count

    text = re.sub(r'([,])', r'\1…', text)
    text = re.sub(r'([.?!])', r'\1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\b(Mas|Porém|Entretanto|Então|Ou seja|Por isso)\b', r'\1...', text)
    text = re.sub(r'\.\.\.', '…', text)

    # === 2️⃣ Parâmetros de voz (Brian) ===
    stability = 0.55
    similarity_boost = 0.92
    style = 0.45
    use_speaker_boost = True

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost
        },
        "text": text
    }

    # === 3️⃣ Retry com 3 tentativas ===
    tentativas = 3
    for tentativa in range(1, tentativas + 1):
        try:
            print(f"🎧 [TTS] Gerando narração (tentativa {tentativa}/{tentativas}, voz Brian, speed={speed})…")
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
            resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=90)

            if resp.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(1024):
                        if chunk:
                            f.write(chunk)

                # === 4️⃣ Ajuste fino da velocidade ===
                if abs(speed - 1.0) > 0.02:
                    temp = str(out_path) + "_tmp.mp3"
                    os.rename(out_path, temp)
                    subprocess.run([
                        "ffmpeg", "-y", "-i", temp, "-filter:a", f"atempo={speed}", out_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.remove(temp)

                print(f"✅ Narração salva com voz Brian ({speed}x) em {out_path}")
                return out_path  # sucesso → sai da função

            else:
                print(f"⚠️ Erro HTTP {resp.status_code} → {resp.text[:100]}")
                time.sleep(5)

        except Exception as e:
            print(f"⚠️ Tentativa {tentativa}/{tentativas} falhou ({e}), aguardando 5s e tentando novamente…")
            time.sleep(5)

    # === 5️⃣ Fallback final (todas tentativas falharam) ===
    print(f"❌ Falha após {tentativas} tentativas. Salvando texto para debug.")
    with open(str(out_path) + ".txt", "w", encoding="utf-8") as f:
        f.write(f"[TTS FAIL – Brian]\n\n{text}")
    time.sleep(1)



def make_intro_card(bg_image: Path, audio_intro: Path, out_video: Path, w=1920, h=1080):
    """Card de introdução simples (blur + título) usando duração do áudio."""
    dur = max(2.5, ffprobe_duration(audio_intro))
    is_windows = (os.name == "nt")
    font_path = get_system_font().replace("\\", "/")
    font_opt = "font='Arial'" if is_windows else f"fontfile='{font_path}'"
    txt = "Hoje: três checagens rápidas"
    def _esc(s): return s.replace("\\","\\\\").replace("'", r"\'").replace(":","\\:").replace(",","\\,").replace("=","\\=")
    txt = _esc(txt)
    filt = (
        f"[0:v]scale={w}:{h},boxblur=luma_radius=12:luma_power=1[bg];"
        f"[bg]drawbox=x=(iw-0.70*iw)/2:y=(ih-0.22*ih)/2:w=0.70*iw:h=0.22*ih:color=black@0.45:t=fill[v1];"
        f"[v1]drawtext=text='{txt}':{font_opt}:fontsize=60:fontcolor=white:"
        f"borderw=2:bordercolor=black@0.6:x=(w-text_w)/2:y=(h-text_h)/2[vout]"
    )
    cmd = [
        "ffmpeg","-y",
        "-loop","1","-t", f"{dur:.2f}", "-i", str(bg_image),
        "-i", str(audio_intro),
        "-filter_complex", filt,
        "-map","[vout]","-map","1:a",
        "-c:v","libx264","-preset","ultrafast","-crf","20",
        "-c:a","aac","-b:a","192k","-ar","48000",
        "-shortest","-movflags","+faststart",
        str(out_video)
    ]
    print("▶ Gerando card de introdução…"); run(cmd); print(f"✅ Intro salva: {out_video}")

def normalize_video(input_path: Path, output_path: Path):
    """
    Normaliza fps, SAR e codec para evitar tela preta ao concatenar vídeos.
    Usa formato seguro: yuv420p + 30fps + SAR=1.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", "fps=30,scale=1920:1080,setsar=1",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path)
    ]
    run(cmd)
    return output_path

def concat_many(inputs: list, out_path: Path):
    """
    Concatena vídeos (todos com vídeo e áudio, criando silêncio quando necessário)
    usando o método de lista segura do FFmpeg (-f concat -safe 0 -i list.txt).
    É mais compatível e evita erros de 'matches no streams' em Windows.
    """
    if not inputs:
        print("⚠️ Nada para concatenar (lista vazia).")
        return

    safe_inputs = []
    for i, f in enumerate(inputs):
        p = Path(f)
        if not p.exists():
            print(f"⚠️ Arquivo ausente: {p}")
            continue

        # Detecta se há áudio
        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0", str(p)
        ], capture_output=True, text=True)
        if not probe.stdout.strip():
            # Sem áudio: cria uma versão com áudio silencioso
            silent = p.parent / f"tmp_silence_{i:02d}.mp4"
            dur = ffprobe_duration(p)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(p),
                "-f", "lavfi", "-t", f"{dur:.2f}",
                "-i", "anullsrc=r=48000:cl=stereo",
                "-shortest",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                str(silent)
            ]
            run(cmd)
            safe_inputs.append(silent)
        else:
            safe_inputs.append(p)

    # Cria lista temporária
    list_file = out_path.parent / "concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for v in safe_inputs:
            f.write(f"file '{v.as_posix()}'\n")

    # Concat via lista
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path)
    ]

    print("🎬 Concatenando clipes em modo seguro (lista concat)…")
    run(cmd)
    print(f"✅ Vídeo final concatenado com sucesso: {out_path}")

# ============================== THUMBNAIL (OPENAI) =========================
def gerar_thumbnail_openai_dinamica(titulo_video: str, resumo: str, veredito: str, out_path: Path):
    """
    Gera uma thumbnail com OpenAI Images (DALL·E / gpt-image-1), 1920x1080.
    """
    try:
        client = openai_images_client()
        prompt = (
            f"Crie uma thumbnail de YouTube 16:9 (1920x1080) para o vídeo '{titulo_video}'. "
            f"Tema: {resumo}. Veredito predominante: '{veredito}'. "
            "Incluir um apresentador/detetive em cena de investigação, com elementos de debate. "
            "Adicione UM texto grande (até 5 palavras), legível em mobile, coerente com título/veredito "
            f"(ex.: '{veredito.upper()}?', 'VERDADE OU MITO?', 'CHINA DOMINA?'). "
            "Estilo cinematográfico, alto contraste, profissional, cores vibrantes. "
            "Alta legibilidade, evitar poluição visual e parágrafos longos."
        )

        print("🎨 Gerando thumbnail via OpenAI…")
        result = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size="1536x1024",  # thumbnail horizontal 16:9 aproximado
            n=1
        )
        image_b64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_b64)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        print(f"✅ Thumbnail salva: {out_path}")
        return out_path
    except Exception as e:
        print(f"⚠️ Erro ao gerar thumbnail com OpenAI: {e}")
        return None

# ============================== CONCAT (shorts) ============================
def concat_three(short_video: Path, vignette: Path, verdict: Path, out_path: Path):
    ins = []; parts = []; idx = 0
    for f in [short_video, vignette, verdict]:
        if f and Path(f).exists():
            ins += ["-i", str(f)]
            parts.append(f"[{idx}:v][{idx}:a]"); idx += 1
    if not parts:
        print("⚠️  Nada para concatenar. Pulando."); return
    filt = "".join(parts) + f"concat=n={len(parts)}:v=1:a=1[v][a]"
    cmd = ["ffmpeg","-y", *ins, "-filter_complex", filt, "-map","[v]","-map","[a]",
           "-c:v","libx264","-preset","ultrafast","-crf","20","-c:a","aac","-b:a","192k","-ar","48000",
           "-movflags","+faststart", str(out_path)]
    run(cmd); print(f"🎞️ Short final: {out_path.name}")

# ============================== PIPELINE LONGO =============================
def build_long_video(video_base: Path, cortes_fc: list, w: int, h: int, transcricao_txt: str):
    """
    Versão FINAL e funcional:
    - Usa filter_complex concat (sem travar vídeo)
    - Corrige erro de filtros simultâneos
    - Mantém áudio sincronizado e vídeo fluido
    """
    LONG_DIR.mkdir(exist_ok=True)

    # 1️⃣ Gera intro narrada
    intro_mp3 = OUT_DIR / "intro_long.mp3"
    # Use as 3 primeiras narrações em estilo YouTube para montar uma intro fluida
    intro_texto = "Hoje: três checagens rápidas — " + " ".join(
        [c.get("narracao_youtube", "") for c in cortes_fc[:3]]
    )
    tts_elevenlabs(intro_texto, intro_mp3)

    intro_bg = OUT_DIR / "intro_bg.png"
    run(["ffmpeg", "-y", "-i", str(video_base), "-frames:v", "1", str(intro_bg)])

    intro_mp4 = LONG_DIR / "intro.mp4"
    make_intro_card(intro_bg, intro_mp3, intro_mp4, w=w, h=h)
    print(f"✅ Intro salva: {intro_mp4}")

    # 2️⃣ Monta lista de vídeos (intro + shorts)
    shorts = sorted(SHORTS_DIR.glob("short_*_final.mp4"))
    if not shorts:
        print("⚠️ Nenhum short encontrado em shorts_final/.")
        return None

    inputs = [intro_mp4] + shorts
    final_long = LONG_DIR / "video_padrao_final.mp4"

    # 3️⃣ Concatenação com filter_complex
    cmd = ["ffmpeg", "-y"]
    for inp in inputs:
        cmd += ["-i", str(inp)]

    n = len(inputs)

    # Cada entrada: [0:v][0:a][1:v][1:a]... concat=n={n}:v=1:a=1[v][a]
    filter_concat = "".join(f"[{i}:v][{i}:a]" for i in range(n))
    # aplica fps/format/setsar dentro do mesmo grafo
    filter_concat += f"concat=n={n}:v=1:a=1[v0][a0];[v0]fps=30,format=yuv420p,setsar=1[v]"

    cmd += [
        "-filter_complex", filter_concat,
        "-map", "[v]", "-map", "[a0]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(final_long)
    ]

    print(f"🎬 Concatenando {n} vídeos com filter_complex concat…")
    run(cmd)
    print(f"✅ Vídeo longo salvo: {final_long}")

    # 4️⃣ Thumbnail
    resumo_thumb = resumir_conteudo_video(transcricao_txt)
    ordem = ["Verdadeiro", "Impreciso", "Falso", "Sem dados suficientes", "Sem dados"]
    try:
        veredito_dominante = max(
            [c.get("checagem", {}).get("status", "Sem dados suficientes") for c in cortes_fc[:3]],
            key=lambda v: ordem.index(v) if v in ordem else len(ordem)
        )
    except Exception:
        veredito_dominante = "Verificado"

    thumb_path = LONG_DIR / "thumbnail.png"
    gerar_thumbnail_openai_dinamica(
        titulo_video="Cortes Verificados",
        resumo=resumo_thumb,
        veredito=veredito_dominante,
        out_path=thumb_path
    )

    print("✅ Vídeo longo finalizado com sucesso e sem congelamentos.")
    return final_long

def calcular_custo_estimado():
    """
    Exibe o custo total estimado em USD e BRL com base no uso da OpenAI e ElevenLabs.
    Mostra tokens e caracteres detalhados.
    """
    # Preços atuais (novembro/2025)
    # GPT-4o
    gpt4o_prompt_usd = 0.0025 / 1000
    gpt4o_output_usd = 0.0100 / 1000
    # GPT-4o-mini (caso usado em metadados)
    gpt4o_mini_prompt_usd = 0.00015 / 1000
    gpt4o_mini_output_usd = 0.00060 / 1000
    # ElevenLabs (plano padrão – voz Brian)
    elevenlabs_char_usd = 0.0004 / 1000  # por 1k caracteres

    # Cálculo OpenAI
    openai_prompt_cost = openai_stats["prompt_tokens"] * gpt4o_prompt_usd
    openai_completion_cost = openai_stats["completion_tokens"] * gpt4o_output_usd
    openai_total_usd = openai_prompt_cost + openai_completion_cost

    # Cálculo ElevenLabs
    eleven_total_usd = eleven_stats["characters"] * elevenlabs_char_usd

    # Totais combinados
    total_usd = openai_total_usd + eleven_total_usd
    total_brl = total_usd * 5.7  # câmbio aproximado

    print("\n" + "=" * 60)
    print("💰 RELATÓRIO FINAL DE CUSTOS ESTIMADOS")
    print("=" * 60)
    print(f"🧠 OpenAI (GPT-4o):")
    print(f"   • Requisições:        {openai_stats['requests']}")
    print(f"   • Tokens prompt:      {openai_stats['prompt_tokens']:,}")
    print(f"   • Tokens resposta:    {openai_stats['completion_tokens']:,}")
    print(f"   • Total tokens:       {openai_stats['total_tokens']:,}")
    print(f"   • Custo estimado:     ${openai_total_usd:.4f} USD (~R${openai_total_usd * 5.7:.2f})")

    print(f"\n🎙️ ElevenLabs (voz Brian):")
    print(f"   • Requisições:        {eleven_stats['requests']}")
    print(f"   • Caracteres usados:  {eleven_stats['characters']:,}")
    print(f"   • Custo estimado:     ${eleven_total_usd:.4f} USD (~R${eleven_total_usd * 5.7:.2f})")

    print("\n" + "-" * 60)
    print(f"💵 TOTAL ESTIMADO:       ${total_usd:.4f} USD  (~R${total_brl:.2f})")
    print("=" * 60 + "\n")

    return total_usd, total_brl


from youtube_auth import get_youtube_service

from googleapiclient.http import MediaFileUpload

def upload_to_youtube(video_path, title, description, tags, thumbnail_path):
    youtube = get_youtube_service()

    request_body = {
        "snippet": {
            "categoryId": "25",  # Notícias e política
            "title": title,
            "description": description,
            "tags": tags
        },
        "status": {"privacyStatus": "public"}
    }

    print("📤 Enviando vídeo para o YouTube...")

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"⬆️  Upload em {int(status.progress() * 100)}%...")
    print(f"✅ Vídeo enviado com sucesso! ID: {response['id']}")

    if thumbnail_path and os.path.exists(thumbnail_path):
        youtube.thumbnails().set(
            videoId=response["id"],
            media_body=thumbnail_path
        ).execute()
        print("🖼️ Thumbnail enviada com sucesso.")

    return response["id"]

def gerar_metadados_youtube(tipo, video_titulo_original, contexto, YOUTUBE_URL, client):
    """
    Gera metadados otimizados para YouTube (título, descrição, tags)
    com fallback inteligente e logs detalhados.
    """
    print("\n" + "="*80)
    print(f"🧠 GERANDO METADADOS ({tipo.upper()}) PARA YOUTUBE")
    print("="*80)

    def log_debug(data):
        """Loga tanto no console quanto em arquivo JSONL"""
        try:
            log_path = Path("saida/debug_metadata_log.jsonl")
            log_path.parent.mkdir(exist_ok=True, parents=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            print("🪵 [DEBUG]", json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"⚠️ Falha ao salvar log de debug: {e}")

    # ------------------- ETAPA 1: DETECÇÃO DE PARTICIPANTES/PODCAST -------------------
    podcast_nome, participantes, tema = None, [], None

    try:
        detection_prompt = f"""
Analise o título e a transcrição abaixo e identifique:
1. Nome do podcast, se houver (Flow, Inteligência Ltda, Podpah, etc.)
2. Nome dos principais participantes (apresentador e convidados)
3. Tema central discutido (resuma em até 5 palavras)

Retorne APENAS um JSON válido no formato:
{{
  "podcast": "NOME DO PODCAST OU null",
  "participantes": ["NOME1", "NOME2"],
  "tema": "TEMA CENTRAL"
}}

Título original: "{video_titulo_original}"
Trecho/contexto: {contexto[:800]}
"""
        resp = openai_request_with_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um especialista em SEO e otimização de vídeos para YouTube."},
                {"role": "user", "content": detection_prompt},
            ],
            temperature=0.8,
        )

        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip("`\n ")

        metadata = json.loads(raw)
        podcast_nome = metadata.get("podcast")
        participantes = metadata.get("participantes") or []
        tema = metadata.get("tema")

        log_debug({"etapa": "detecção", "podcast": podcast_nome, "participantes": participantes, "tema": tema})
        print("🎧 Podcast detectado:", podcast_nome or "N/D")
        print("🎙️ Participantes:", ", ".join(participantes) if participantes else "N/D")
        print("🗣️ Tema:", tema or "N/D")

    except Exception as e:
        log_debug({"etapa": "detecção", "erro": str(e)})
        print(f"⚠️ Erro ao detectar metadados: {e}")
        podcast_nome, participantes, tema = None, [], None

    # ------------------- ETAPA 2: GERAÇÃO SEO -------------------
    seo_prompt = f"""
Gere TÍTULO, DESCRIÇÃO e TAGS para YouTube com base nas informações abaixo,
**OBRIGATORIAMENTE incluindo o nome do podcast (se houver) e dos participantes principais no TÍTULO e na DESCRIÇÃO**.

Tipo de vídeo: {tipo.upper()}
URL: {YOUTUBE_URL}
Título original: "{video_titulo_original}"
Podcast: {podcast_nome or 'Desconhecido'}
Participantes: {', '.join(participantes) or 'Não identificado'}
Tema: {tema or 'Não informado'}
Contexto adicional: {contexto[:600]}

### REGRAS:
1. O TÍTULO deve conter:
   - O nome do apresentador e dos principais convidados
   - O nome do podcast (se existir)
   - Um verbo de ação forte (ex: debate, confronta, reage, explica)
   - 65–95 caracteres
   - Use emoção e curiosidade autênticas (sem clickbait vazio); verbos fortes como revela, explica, confronta, surpreende.
   - Gere um título com emoção e curiosidade autênticas (sem clickbait vazio), usando verbos fortes (revela, explica, confronta, surpreende). 65–90 caracteres.
   - Sem ponto final.
   - Se houver uma pessoa notória (convidado famoso, personalidade ou nome reconhecido), o título deve COMEÇAR com esse nome. 
     Exemplo: "🚀 Ramon Dino REVELA Verdade no Flow Podcast"
   - Reforce: o título deve SEMPRE começar com o nome mais famoso ou reconhecido quando existir, para maximizar CTR.
   - Priorize títulos curtos e impactantes (até 70 caracteres) que funcionem bem em vídeos de até 40 segundos.




2. A DESCRIÇÃO deve:
   - Começar com o link do vídeo original.
   - Nas 2 primeiras linhas, repetir os nomes e o tema.
   - Incluir chamada à ação (Inscreva-se, Comente, etc.)
   - Finalizar com hashtags.

3. As TAGS devem conter de 10 a 20 palavras-chave separadas por vírgulas,
incluindo os nomes dos participantes e o podcast.

Retorne APENAS um JSON válido:
{{
  "title": "TÍTULO OTIMIZADO",
  "description": "DESCRIÇÃO OTIMIZADA",
  "tags": "tag1, tag2, tag3"
}}
"""

    try:
        print("\n🧩 Gerando SEO (título, descrição e tags)...")
        resp = openai_request_with_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um especialista em SEO e otimização de vídeos para YouTube."},
                {"role": "user", "content": seo_prompt},
            ],
            temperature=0.8,
        )

        raw = (resp.choices[0].message.content or "").strip()
        log_debug({"etapa": "seo", "raw": raw})

        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip("`\n ")

        if not raw or not raw.startswith("{"):
            raise ValueError("Resposta vazia ou inválida do GPT ao gerar SEO.")

        metadata = json.loads(raw)
        title = metadata.get("title") or f"{video_titulo_original} | Cortes Verificados"

        # Garante que a marca "Fact-Check" sempre apareça no título
        if "fact-check" not in title.lower():
            # Adiciona no início, mantendo o estilo
            title = f"Fact-Check: {title.strip()}"

        # --- Anexar nome do podcast ao título (SEO e coerência visual) ---
        podcast_name = detectar_nome_podcast(video_titulo_original)

        # Evita repetir se o GPT já inseriu o nome no título
        if podcast_name.lower() not in title.lower():
            # Coloca um separador visual — usa "|" para shorts e "-" para vídeos longos
            separador = "|" if tipo == "short" else "-"
            title = f"{title.strip()} {separador} {podcast_name}"

        # --- Emoji no início do título ---==
        if ENABLE_TITLE_EMOJI:
            # evita duplicar se já vier com emoji
            starts_with_emoji = any(
                title.startswith(prefix) for prefix in ("🎙️", "🧠", "⚡", "🔥", "🚀", "🤖", "🧪", "📈", "🏛️", "✅", "❗", "⭐")
            )
            if not starts_with_emoji:
                # usa o contexto (descrição/tema detectado) + tipo (short/long) para escolher o emoji
                # se sua função recebe "tipo" como parâmetro ("short" / "long"), passe-o aqui
                emoji = escolher_emoji_para_titulo(contexto=contexto or "",
                                                   tipo=tipo if tipo in ("short", "long") else "short")

                # evita estourar limite — só aplica se couber
                # (se seu prompt já limita 65–90, isso quase sempre cabe)
                if len(title) <= 98:  # 1 emoji + espaço
                    title = f"{emoji} {title}"

        desc = metadata.get("description") or f"Veja aqui o vídeo completo: {YOUTUBE_URL}\n\n{contexto}"
        if "https://" in desc and not desc.strip().lower().startswith("veja aqui o vídeo completo:"):
            desc = f"Veja aqui o vídeo completo: {YOUTUBE_URL}\n" + desc

        # CTAs textuais (engajamento)
        cta_block = (
            "\n\n💬 Comente o que achou deste corte!\n"
            "📢 Inscreva-se para mais vídeos como este!\n"
            "🔔 Ative o sininho para não perder os próximos cortes verificados."
        )
        if cta_block.strip() not in desc:
            desc += cta_block

        # --- Bloco legal de crédito e uso transformativo ---
        podcast_name = detectar_nome_podcast(video_titulo_original)
        disclaimer_block = (
            f"\n\n🎙️ Trecho retirado do {podcast_name}.\n"
            f"📺 Vídeo original: {YOUTUBE_URL}\n"
            "💡 Este canal realiza checagem de fatos e comentários educativos sobre os trechos, "
            "caracterizando uso transformativo conforme diretrizes de fair use."
        )

        if disclaimer_block.strip() not in desc:
            desc += disclaimer_block

        # Tags geradas pelo GPT (string) → lista
        tags_from_gpt = (metadata.get("tags") or "cortes verificados, fact check, podcast").strip()
        tags_list = [t.strip() for t in tags_from_gpt.split(",") if t.strip()]

        # Tags fixas do canal (consistência de nicho)
        default_tags = [
            "cortes verificados", "fact check", "podcast", "Flow Podcast",
            "YouTube Shorts", "ciência", "tecnologia", "debate", "veredito"
        ]

        # Mescla sem duplicar (case-insensitive)
        existing_lower = {t.lower() for t in tags_list}
        for t in default_tags:
            if t.lower() not in existing_lower:
                tags_list.append(t)

        tags = tags_list  # ← **retorne lista**; seu uploader aceita lista

        print(f"\n🧠 [{tipo.upper()}] Título gerado:\n   {title}")
        print(f"📝 Descrição (primeiras linhas):\n   {desc[:250]}{'...' if len(desc) > 250 else ''}")
        print(f"🏷️ Tags:\n   {tags}")
        print("="*80)

        return title, desc, tags

    except Exception as e:
        print(f"⚠️ Erro ao gerar metadados YouTube ({tipo}): {e}")
        log_debug({"etapa": "seo", "erro": str(e), "prompt": seo_prompt[:300]})

        # 🔄 FALLBACK mais inteligente para SHORTS individuais
        if tipo.lower().startswith("short"):
            base_title = video_titulo_original.split(" - ")[0]
            fallback_title = f"{base_title} | {tema or 'Corte Verificado'}"
            fallback_desc = f"Assista ao vídeo original: {YOUTUBE_URL}\n\n{contexto[:200]}..."
            fallback_tags = "cortes verificados, fact check, podcast, política, curiosidades"
            return fallback_title, fallback_desc, fallback_tags
        else:
            return f"{video_titulo_original} | Cortes Verificados", f"Assista ao vídeo original: {YOUTUBE_URL}", "cortes verificados, fact check, podcast"





# ============================== MAIN ======================================
def main():
    global ELEVEN_API_KEY

    ELEVEN_API_KEY = escolher_melhor_elevenlabs_key()

    # 0) Preparação
    if not OPENAI_API_KEY or "COLOQUE_SUA_OPENAI_KEY_AQUI" in OPENAI_API_KEY:
        print("⚠️ Defina OPENAI_API_KEY no script.")
    if not ELEVEN_API_KEY or "COLOQUE_SUA_ELEVEN_KEY_AQUI" in ELEVEN_API_KEY:
        print("⚠️ Defina ELEVEN_API_KEY no script.")

    clean_dirs()

    # 1) Download
    video_original = download_youtube_video(YOUTUBE_URL, OUT_DIR)

    # 2) Extração de áudio sem correção (usaremos vídeo original para cortes)
    wav_16k = OUT_DIR / (video_original.stem + "_mod.wav")
    extract_wav_mono16k(video_original, wav_16k)
    video_mod = video_original  # mantém referência

    # 3) Transcrição + Diarização + Merge
    asr = transcribe_whisper(wav_16k, WHISPER_SIZE)
    diar = diarize_local(wav_16k)
    merged = assign_speakers(asr, diar)
    txt_path = OUT_DIR / (video_mod.stem + "_transcricao.txt")
    write_transcript(merged, txt_path)

    # 4) Texto “plano” para GPT
    transcricao_txt = build_context_from_segments(merged)

    # 5) Cortes (candidatos) com GPT
    cortes_candidatos = gpt_sugerir_cortes(merged)
    top_cortes = filtra_e_otimiza_topN(cortes_candidatos, TOP_N_CORTES, SHORT_MIN_S, SHORT_MAX_S)

    # 6) Fact-check por corte
    top_cortes_fc = fact_check_cortes(top_cortes, segments=merged)

    # Salva JSON consolidado (fica em OUT_DIR, será limpo no final)
    json_out = OUT_DIR / "shorts_top10_verificados.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(top_cortes_fc, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON consolidado: {json_out}")

    # 7) Vinheta (16:9)
    w, h = ffprobe_resolution(video_mod)
    vignette = OUT_DIR / "vinheta.mp4"
    make_vignette(vignette, w, h)

    # 8) Shorts (gera final 16:9 e vertical 9:16) — intermediários ficam no OUT_DIR
    for idx, c in enumerate(top_cortes_fc, 1):
        print(f"\n▶ Gerando short {idx}…")
        # 8.1) Recorte do vídeo mod
        short_raw = OUT_DIR / f"short_{idx:02d}.mp4"
        # 8.1) Recorte do vídeo mod
        short_raw = OUT_DIR / f"short_{idx:02d}.mp4"
        result = cut_short(video_mod, c["inicio"], c["fim"], short_raw)

        # Aplica correção sutil de imagem apenas no corte, não no vídeo inteiro
        short_corrected = OUT_DIR / f"short_{idx:02d}_corrigido.mp4"
        hflip_eq_pitch_cut(short_raw, short_corrected)

        if result is None or not short_corrected.exists():
            print(f"⚠️ Short {idx} inválido (sem vídeo gerado). Pulando…")
            continue

        dur = ffprobe_duration(short_corrected)
        if dur <= 0.5:
            print(f"⚠️ Short {idx} muito curto ({dur:.2f}s). Pulando…")
            continue

        # 8.2) Overlays
        short_overlay = OUT_DIR / f"short_{idx:02d}_overlay.mp4"
        overlay_logo_and_detective(short_corrected, LOGO_PATH, DETECTIVE_PNG, short_overlay, w)

        # 8.3) Fade-out no final (limpar transição p/ vinheta)
        short_fade = OUT_DIR / f"short_{idx:02d}_fade.mp4"
        fade_out_audio(short_overlay, short_fade, fade_s=0.35)

        # 8.4) Veredito narrado
        # Extrai diretamente do fact_check_cortes()
        status = c.get("status", "Sem dados suficientes").strip().capitalize()
        fonte = c.get("fonte") or "N/A"
        resumo = c.get("resumo") or ""
        nar_text = c.get("narracao_youtube") or c.get("narracao") or (
            f"A gente conferiu rapidinho. {resumo if resumo else 'Não há dados sólidos sobre isso.'} "
            f"Veredito: {status}. Fonte: {fonte if fonte and fonte != 'N/A' else 'sem fonte confiável'}."
        )

        narr_mp3 = OUT_DIR / f"narr_{idx:02d}.mp3"
        tts_elevenlabs(nar_text.strip(), narr_mp3)

        # 8.5) Último frame do short
        last_png = OUT_DIR / f"last_{idx:02d}.png"
        make_last_frame(short_fade, last_png)

        # 8.6) Card de veredito (16:9)
        verdict_mp4 = OUT_DIR / f"veredito_{idx:02d}.mp4"
        make_verdict_card(
            bg_image=last_png,
            detective_png=DETECTIVE_PNG,
            narration_audio=narr_mp3,
            verdict=status,
            fonte=fonte,
            w=w, h=h,
            out_video=verdict_mp4
        )

        # 8.7) Concat short16:9 -> vinheta -> veredito16:9 (resultado final vai para SHORTS_DIR)
        SHORTS_DIR.mkdir(exist_ok=True)
        final_mp4 = SHORTS_DIR / f"short_{idx:02d}_final.mp4"
        concat_three(short_fade, vignette, verdict_mp4, final_mp4)

        # 8.8) Versão vertical 9:16 do short final
        final_vertical = SHORTS_DIR / f"short_{idx:02d}_final_vertical.mp4"
        make_vertical_from_16x9(final_mp4, final_vertical)

    # 9) Vídeo padrão (16:9): intro narrada + 3 cortes + vereditos + thumbnail
    # 9) Vídeo padrão (16:9): intro narrada + 3 cortes + vereditos + thumbnail
    GERAR_VIDEO_LONGO = False
    if GERAR_VIDEO_LONGO:
        final_long = build_long_video(video_mod, top_cortes_fc, w, h, transcricao_txt)
        if final_long:
            print(f"✅ Vídeo padrão gerado em: {final_long}")

    # 🔥 10) LIMPEZA FINAL: mantém apenas shorts_final/*, long_final/video_padrao_final.mp4 e long_final/thumbnail.png
    try:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        # Garante que apenas finais estão nas pastas:
        # (Como todos intermediários foram escritos em OUT_DIR, isto já resolve.)
        print("🧹 Limpeza concluída. Mantidos apenas os arquivos finais.")
    except Exception as e:
        print(f"⚠️ Erro ao limpar intermediários: {e}")

    print("\n✅ Pipeline concluído com sucesso! 🎉")
    print(f"• Shorts finais (16:9 e 9:16): {SHORTS_DIR}")
    print(f"• Vídeo padrão 16:9 e thumbnail: {LONG_DIR}")

    # === 11) Metadados para vídeos ===
    # === 11) Metadados para vídeos ===
    print("🧠 Gerando metadados para vídeos...")

    # Detecta automaticamente o título original do vídeo baixado
    video_titulo_original = Path(video_original).stem

    # Metadados para vídeo longo (16:9)
    client = openai_client()

    title_long, desc_long, tags_long = gerar_metadados_youtube(
        tipo="long",
        video_titulo_original=video_titulo_original,
        contexto="Resumo geral do vídeo completo com todas as checagens e vereditos.",
        YOUTUBE_URL=YOUTUBE_URL,
        client=client
    )

    # Metadados para os shorts (9:16)
    title_short, desc_short, tags_short = gerar_metadados_youtube(
        tipo="short",
        video_titulo_original=video_titulo_original,
        contexto="Cortes curtos de fact-check baseados no vídeo principal.",
        YOUTUBE_URL=YOUTUBE_URL,
        client=client
    )

    # === 12) Upload ===
    if GERAR_VIDEO_LONGO:
        video_path_long = str(LONG_DIR / "video_padrao_final.mp4")
        thumb_path = str(LONG_DIR / "thumbnail.png")
        try:
            upload_to_youtube(video_path_long, title_long, desc_long, tags_long, thumb_path)
        except Exception as e:
            print(f"⚠️ Erro ao subir vídeo longo: {e}")

    # Agora sobe cada short com metadados exclusivos
    short_files = sorted(SHORTS_DIR.glob("*_vertical.mp4"))

    for idx, short_file in enumerate(short_files, start=1):
        print(f"\n🎬 Gerando metadados exclusivos para short {idx}: {short_file.name}")
        try:
            # Gerar metadados individuais com contexto do corte correspondente
            corte = top_cortes_fc[idx - 1] if idx - 1 < len(top_cortes_fc) else None
            # 🔹 Contexto detalhado para gerar títulos variados
            contexto = ""
            if corte:
                narracao = corte.get("narracao_youtube") or corte.get("narracao") or ""
                veredito = corte.get("status", "N/A")
                fonte = corte.get("fonte", "N/A")
                titulo_corte = corte.get("titulo", "")
                resumo = corte.get("descricao", "") or corte.get("resumo", "")

                contexto = (
                    f"Título do corte: {titulo_corte}. "
                    f"Resumo: {resumo}. "
                    f"Narração: {narracao}. "
                    f"Veredito: {veredito}. "
                    f"Fonte: {fonte}. "
                    f"Este é um corte autônomo do vídeo principal."
                )

            title_s, desc_s, tags_s = gerar_metadados_youtube(
                tipo="short",
                video_titulo_original=video_titulo_original,
                contexto=contexto,
                YOUTUBE_URL=YOUTUBE_URL,
                client=client
            )

            upload_to_youtube(short_file, title_s, desc_s, tags_s, None)

            # Upload também para o Facebook
            try:
                title_fb, desc_fb, hashtags_fb = gerar_metadados_facebook(title_s, desc_s, tags_s, client)
                upload_to_facebook(short_file, title_fb, f"{desc_fb}\n{' '.join(hashtags_fb)}")
            except Exception as e_fb:
                print(f"⚠️ Erro ao subir vídeo no Facebook: {e_fb}")


        except Exception as e:
            print(f"⚠️ Erro ao subir short {idx} ({short_file.name}): {e}")


if __name__ == "__main__":
    try:
        main()
    finally:
        calcular_custo_estimado()
