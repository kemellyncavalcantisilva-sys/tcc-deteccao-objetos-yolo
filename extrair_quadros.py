"""
extrair_quadros.py - Extrai quadros dos vídeos de teste para anotação manual.

Pega quadros igualmente espaçados de cada vídeo da pasta, para montar o
conjunto rotulado usado na avaliação de precisão (mAP).

Uso:
  python extrair_quadros.py
  python extrair_quadros.py --pasta videos --por-video 25 --saida quadros_anotacao
"""

import argparse
import glob
import os

import cv2

p = argparse.ArgumentParser(description="Extração de quadros para anotação")
p.add_argument("--pasta", default="videos", help="pasta com os vídeos .mp4")
p.add_argument("--por-video", type=int, default=25, help="quantidade de quadros por vídeo")
p.add_argument("--saida", default="quadros_anotacao")
args = p.parse_args()

videos = sorted(glob.glob(os.path.join(args.pasta, "*.mp4")))
if not videos:
    raise SystemExit(f"Nenhum vídeo .mp4 encontrado na pasta '{args.pasta}'.")

os.makedirs(args.saida, exist_ok=True)
total = 0
for caminho in videos:
    nome = os.path.splitext(os.path.basename(caminho))[0]
    cap = cv2.VideoCapture(caminho)
    n_quadros = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_quadros <= 0:
        print(f"{nome}: não foi possível ler o vídeo")
        continue
    passo = max(n_quadros // args.por_video, 1)
    salvos = 0
    for indice in range(0, n_quadros, passo):
        if salvos >= args.por_video:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(os.path.join(args.saida, f"{nome}_{indice:04d}.jpg"), frame)
        salvos += 1
    cap.release()
    total += salvos
    print(f"{nome}: {salvos} quadros extraídos")

print(f"\nTotal: {total} imagens em '{args.saida}'. Envie essa pasta para o Roboflow.")
