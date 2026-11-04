"""
anotar.py - Anotador simples de imagens para a avaliação de precisão (TCC).

Abre cada imagem da pasta de entrada; você desenha caixas com o mouse e escolhe a
classe pelo teclado. As anotações são salvas no formato YOLO, já com a numeração
das classes do conjunto COCO, prontas para o avaliar_precisao.py.

Uso:
  python anotar.py
  python anotar.py --entrada quadros_anotacao --saida dataset_coco

Controles:
  1 a 8          escolhe a classe da próxima caixa
  mouse          clique e arraste para desenhar a caixa
  Z              desfaz a última caixa
  ENTER/ESPAÇO   salva e vai para a próxima imagem
  A              salva e volta para a imagem anterior
  Q ou ESC       salva e encerra (dá para continuar depois de onde parou)
"""

import argparse
import glob
import os
import shutil

import cv2
import yaml

# tecla: (nome da classe, índice no COCO)
CLASSES = {
    1: ("person", 0), 2: ("cell phone", 67), 3: ("bottle", 39), 4: ("cup", 41),
    5: ("book", 73), 6: ("scissors", 76), 7: ("remote", 65), 8: ("laptop", 63),
}
CORES = {1: (255, 80, 0), 2: (0, 220, 255), 3: (0, 200, 0), 4: (0, 0, 255),
         5: (200, 0, 200), 6: (255, 0, 255), 7: (0, 140, 255), 8: (255, 255, 0)}
COCO_PARA_TECLA = {coco: tecla for tecla, (_, coco) in CLASSES.items()}
ESCALA = 1.5  # amplia a imagem na tela para facilitar o desenho
JANELA = "Anotador - TCC"

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"]


def caminho_rotulo(pasta_lbl, img):
    return os.path.join(pasta_lbl, os.path.splitext(os.path.basename(img))[0] + ".txt")


def carregar(pasta_lbl, img, w, h):
    caixas = []
    arq = caminho_rotulo(pasta_lbl, img)
    if os.path.exists(arq):
        for linha in open(arq, encoding="utf-8"):
            p = linha.split()
            if len(p) == 5 and int(p[0]) in COCO_PARA_TECLA:
                cx, cy, bw, bh = (float(v) for v in p[1:])
                caixas.append((COCO_PARA_TECLA[int(p[0])], int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                               int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return caixas


def salvar(pasta_img, pasta_lbl, img, caixas, w, h):
    destino = os.path.join(pasta_img, os.path.basename(img))
    if not os.path.exists(destino):
        shutil.copy(img, destino)
    with open(caminho_rotulo(pasta_lbl, img), "w", encoding="utf-8") as f:
        for tecla, x1, y1, x2, y2 in caixas:
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            f.write(f"{CLASSES[tecla][1]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")


def main():
    p = argparse.ArgumentParser(description="Anotador simples no formato YOLO/COCO")
    p.add_argument("--entrada", default="quadros_anotacao")
    p.add_argument("--saida", default="dataset_coco")
    args = p.parse_args()

    imagens = sorted(glob.glob(os.path.join(args.entrada, "*.jpg")) + glob.glob(os.path.join(args.entrada, "*.png")))
    if not imagens:
        raise SystemExit(f"Nenhuma imagem encontrada em '{args.entrada}'. Rode antes o extrair_quadros.py.")
    pasta_img = os.path.join(args.saida, "images")
    pasta_lbl = os.path.join(args.saida, "labels")
    os.makedirs(pasta_img, exist_ok=True)
    os.makedirs(pasta_lbl, exist_ok=True)
    with open(os.path.join(args.saida, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": os.path.abspath(args.saida), "train": "images", "val": "images",
                        "names": {i: n for i, n in enumerate(COCO)}}, f, allow_unicode=True, sort_keys=False)

    # começa na primeira imagem ainda não anotada
    idx = next((i for i, im in enumerate(imagens) if not os.path.exists(caminho_rotulo(pasta_lbl, im))), 0)
    estado = {"classe": 1, "inicio": None, "atual": None}

    def mouse(evento, x, y, flags, param):
        x, y = int(x / ESCALA), int(y / ESCALA)
        if evento == cv2.EVENT_LBUTTONDOWN:
            estado["inicio"], estado["atual"] = (x, y), (x, y)
        elif evento == cv2.EVENT_MOUSEMOVE and estado["inicio"]:
            estado["atual"] = (x, y)
        elif evento == cv2.EVENT_LBUTTONUP and estado["inicio"]:
            (x1, y1), (x2, y2) = estado["inicio"], (x, y)
            if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                caixas.append((estado["classe"], min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            estado["inicio"] = estado["atual"] = None

    cv2.namedWindow(JANELA)
    cv2.setMouseCallback(JANELA, mouse)
    while 0 <= idx < len(imagens):
        img_path = imagens[idx]
        base = cv2.imread(img_path)
        h, w = base.shape[:2]
        caixas = carregar(pasta_lbl, img_path, w, h)
        acao = None
        while acao is None:
            tela = cv2.resize(base, None, fx=ESCALA, fy=ESCALA)
            for tecla, x1, y1, x2, y2 in caixas:
                p1, p2 = (int(x1 * ESCALA), int(y1 * ESCALA)), (int(x2 * ESCALA), int(y2 * ESCALA))
                cv2.rectangle(tela, p1, p2, CORES[tecla], 2)
                cv2.putText(tela, CLASSES[tecla][0], (p1[0] + 3, p1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            CORES[tecla], 2)
            if estado["inicio"] and estado["atual"]:
                a = tuple(int(v * ESCALA) for v in estado["inicio"])
                b = tuple(int(v * ESCALA) for v in estado["atual"])
                cv2.rectangle(tela, a, b, CORES[estado["classe"]], 1)
            topo = (f"Imagem {idx + 1}/{len(imagens)}  |  Classe: {estado['classe']} = "
                    f"{CLASSES[estado['classe']][0]}  |  caixas: {len(caixas)}")
            cv2.rectangle(tela, (0, 0), (tela.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(tela, topo, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            legenda = "  ".join(f"{k}:{v[0]}" for k, v in CLASSES.items())
            ajuda = "Z: desfaz  ENTER: proxima  A: anterior  Q: sair"
            cv2.rectangle(tela, (0, tela.shape[0] - 46), (tela.shape[1], tela.shape[0]), (0, 0, 0), -1)
            cv2.putText(tela, legenda, (8, tela.shape[0] - 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(tela, ajuda, (8, tela.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow(JANELA, tela)

            k = cv2.waitKey(20) & 0xFF
            if ord("1") <= k <= ord("8"):
                estado["classe"] = k - ord("0")
            elif k in (ord("z"), ord("Z")) and caixas:
                caixas.pop()
            elif k in (13, 32):
                acao = 1
            elif k in (ord("a"), ord("A")):
                acao = -1
            elif k in (ord("q"), ord("Q"), 27):
                acao = 0
            if cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                acao = 0
        salvar(pasta_img, pasta_lbl, img_path, caixas, w, h)
        if acao == 0:
            break
        idx += acao
        idx = max(idx, 0)

    cv2.destroyAllWindows()
    feitas = sum(os.path.exists(caminho_rotulo(pasta_lbl, im)) for im in imagens)
    print(f"{feitas} de {len(imagens)} imagens anotadas em '{args.saida}'.")
    if feitas == len(imagens):
        print(f"Próximo passo: python avaliar_precisao.py --dados {args.saida}/data.yaml")


if __name__ == "__main__":
    main()
