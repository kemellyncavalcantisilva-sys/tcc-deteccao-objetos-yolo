"""
avaliar_precisao.py - Precisão, Recall e mAP dos modelos YOLO

Precisão só pode ser medida com imagens ROTULADAS (ground truth).
Recebe o arquivo data.yaml de um conjunto anotado no formato YOLO.

Uso:
  python avaliar_precisao.py --dados meu_dataset/data.yaml
"""

import argparse
import csv
import os
from datetime import datetime

from ultralytics import YOLO

p = argparse.ArgumentParser(description="Avaliação de precisão dos modelos YOLO")
p.add_argument("--modelos", nargs="+",
               default=["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"])
p.add_argument("--dados", required=True, help="caminho do data.yaml do conjunto rotulado")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--dispositivo", default="cpu")
args = p.parse_args()


def num(v, casas=4):
    return f"{v:.{casas}f}".replace(".", ",")


os.makedirs("resultados", exist_ok=True)
caminho = os.path.join("resultados", "precisao.csv")
novo = not os.path.exists(caminho)

with open(caminho, "a", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    if novo:
        w.writerow(["data_hora", "modelo", "imgsz", "precisao", "recall",
                    "mAP50", "mAP50_95", "inferencia_ms"])
    for nome in args.modelos:
        print(f"\n=== Avaliando {nome} ===")
        r = YOLO(nome).val(data=args.dados, imgsz=args.imgsz, device=args.dispositivo,
                           batch=4, plots=False, verbose=False)
        linha = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), nome, args.imgsz,
                 num(r.box.mp), num(r.box.mr), num(r.box.map50), num(r.box.map),
                 num(r.speed["inference"], 2)]
        w.writerow(linha)
        print(f"Precisão {linha[3]} | Recall {linha[4]} | mAP50 {linha[5]} | mAP50-95 {linha[6]}")

print(f"\nResultados salvos em {caminho}")
