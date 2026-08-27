import os
import urllib.request
import tarfile
import soundfile as sf

print("=== PLAN B: Bypassing Hugging Face ===")
url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
filename = "dev-clean.tar.gz"

if not os.path.exists(filename):
    print("Descargando corpus original LibriSpeech (337 MB)... esto tomará un momento.")
    urllib.request.urlretrieve(url, filename)
    print("¡Descarga completada!")

print("Extrayendo archivos de audio (esto toma unos segundos)...")
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(path="librispeech_temp")

os.makedirs("audio", exist_ok=True)
groundtruth = []
count = 0

print("Buscando 10 audios que cumplan la rúbrica (15 a 30 segundos)...")
base_dir = "librispeech_temp/LibriSpeech/dev-clean"

for root, dirs, files in os.walk(base_dir):
    if count >= 10: break
    
    for file in files:
        if count >= 10: break
        
        # Buscar los archivos de texto que contienen las transcripciones
        if file.endswith(".txt"):
            with open(os.path.join(root, file), "r") as f:
                lines = f.readlines()
            
            for line in lines:
                if count >= 10: break
                
                # El formato interno es: ID_AUDIO TEXTO_REFERENCIA
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id, text = parts[0], parts[1]
                    flac_path = os.path.join(root, f"{audio_id}.flac")
                    
                    if os.path.exists(flac_path):
                        # Leer el archivo original FLAC
                        data, samplerate = sf.read(flac_path)
                        duration = len(data) / samplerate
                        
                        # Filtro estricto de tiempo
                        if 15.0 <= duration <= 30.0:
                            wav_name = f"test{count+1}.wav"
                            wav_path = os.path.join("audio", wav_name)
                            
                            # Guardar como .wav a 16kHz para Whisper
                            sf.write(wav_path, data, samplerate)
                            
                            # Limpiar el texto para no romper el CSV
                            text_clean = text.replace('"', '')
                            groundtruth.append(f'{wav_name},"{text_clean}"')
                            
                            print(f"[{count+1}/10] Guardado {wav_name} ({duration:.1f}s)")
                            count += 1

# Guardar el documento final
with open("groundtruth.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(groundtruth) + "\n")

print("\n¡Corpus generado con éxito! El archivo groundtruth.txt y la carpeta audio/ están listos.")
