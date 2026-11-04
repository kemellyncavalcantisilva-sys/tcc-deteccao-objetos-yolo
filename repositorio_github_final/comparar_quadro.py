"""
comparar_quadro.py - Compara visualmente as detecções dos modelos em um mesmo quadro.

Lê um quadro de um vídeo de teste, processa com cada modelo e monta uma imagem
em grade (2x2 para 4 modelos), com o nome do modelo e o número de objetos.

Uso:
  python comparar_quadro.py --video videos/pouca_luz.mp4 --saida figura_pouca_luz.jpg
  python comparar_quadro.py --video videos/varios_objetos.mp4 --saida figura_varios_objetos.jpg
  python comparar_quadro.py --video videos/pouca_luz.mp4 --quadro 80 --saida figura_pouca_luz.jpg
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

p = argparse.ArgumentParser(description="Comparação visual das detecções dos modelos")
p.add_argument("--video", required=True, help="vídeo de teste")
p.add_argument("--quadro", type=int, default=-1, help="número do quadro (padrão: o do meio do vídeo)")
p.add_argument("--modelos", nargs="+", default=["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"])
p.add_argument("--saida", default="comparacao.jpg")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--conf", type=float, default=0.25)
args = p.parse_args()

cap = cv2.VideoCapture(args.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n = args.quadro if args.quadro >= 0 else total // 2
cap.set(cv2.CAP_PROP_POS_FRAMES, n)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"Não foi possível ler o quadro {n} de {args.video} (o vídeo tem {total} quadros).")

paineis = []
for nome in args.modelos:
    r = YOLO(nome)(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    img = r.plot(line_width=2)
    titulo = f"{nome.replace('.pt', '').replace('yolo', 'YOLO')}  -  {len(r.boxes)} objetos"
    faixa = np.zeros((44, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(faixa, titulo, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    paineis.append(np.vstack([faixa, img]))
    print(f"{titulo}: {', '.join(r.names[int(c)] for c in r.boxes.cls)}")

if len(paineis) % 2:
    paineis.append(np.full_like(paineis[0], 255))
linhas = [np.hstack(paineis[i:i + 2]) for i in range(0, len(paineis), 2)]
grade = np.vstack(linhas)
cv2.imwrite(args.saida, grade, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"\nImagem salva em {args.saida} (quadro {n} de {total}).")
