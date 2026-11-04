"""
converter_rotulos.py - Prepara o conjunto anotado no Roboflow para a avaliação.

O Roboflow numera as classes na ordem em que foram criadas (0, 1, 2...), mas os
modelos YOLO pré-treinados usam a numeração do conjunto COCO (person = 0,
bottle = 39, cell phone = 67...). Este script converte os rótulos para a
numeração COCO, junta todas as imagens em uma pasta e cria o data.yaml.

Uso:
  python converter_rotulos.py --entrada roboflow_export --saida dataset_coco
Depois:
  python avaliar_precisao.py --dados dataset_coco/data.yaml
"""

import argparse
import glob
import os
import shutil

import yaml

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

# Nomes em português aceitos, caso as classes tenham sido criadas assim no Roboflow
APELIDOS = {"pessoa": "person", "celular": "cell phone", "garrafa": "bottle",
            "cadeira": "chair", "copo": "cup", "caneca": "cup", "notebook": "laptop",
            "teclado": "keyboard", "livro": "book", "taca": "wine glass", "taça": "wine glass",
            "controle": "remote", "controle remoto": "remote", "mochila": "backpack",
            "mesa": "dining table", "televisao": "tv", "televisão": "tv", "monitor": "tv",
            "tesoura": "scissors", "relogio": "clock", "relógio": "clock", "planta": "potted plant",
            "vaso": "vase", "sofa": "couch", "sofá": "couch", "cama": "bed", "gato": "cat",
            "cachorro": "dog", "cellphone": "cell phone", "phone": "cell phone"}


def normalizar(nome):
    nome = str(nome).strip().lower().replace("_", " ").replace("-", " ")
    return APELIDOS.get(nome, nome)


p = argparse.ArgumentParser(description="Conversão de rótulos do Roboflow para a numeração COCO")
p.add_argument("--entrada", required=True, help="pasta exportada do Roboflow (formato YOLO)")
p.add_argument("--saida", default="dataset_coco")
args = p.parse_args()

with open(os.path.join(args.entrada, "data.yaml"), encoding="utf-8") as f:
    nomes = yaml.safe_load(f)["names"]
if isinstance(nomes, dict):
    nomes = [nomes[k] for k in sorted(nomes)]

mapa, erros = {}, []
for i, nome in enumerate(nomes):
    coco = normalizar(nome)
    if coco in COCO:
        mapa[i] = COCO.index(coco)
        print(f"  classe '{nome}' -> COCO {COCO.index(coco)} ({coco})")
    else:
        erros.append(nome)
if erros:
    raise SystemExit(f"\nClasses sem correspondência no COCO: {erros}\n"
                     "Renomeie essas classes no Roboflow usando os nomes em inglês do COCO "
                     "(ex.: 'cell phone', 'bottle', 'person') e exporte de novo.")

pasta_img = os.path.join(args.saida, "images")
pasta_lbl = os.path.join(args.saida, "labels")
os.makedirs(pasta_img, exist_ok=True)
os.makedirs(pasta_lbl, exist_ok=True)

n_imagens, n_objetos = 0, 0
for rotulo in glob.glob(os.path.join(args.entrada, "**", "labels", "*.txt"), recursive=True):
    base = os.path.splitext(os.path.basename(rotulo))[0]
    pasta_imagens = os.path.join(os.path.dirname(os.path.dirname(rotulo)), "images")
    imagens = glob.glob(os.path.join(pasta_imagens, base + ".*"))
    if not imagens:
        continue
    linhas = []
    with open(rotulo, encoding="utf-8") as f:
        for linha in f:
            partes = linha.split()
            if len(partes) == 5:  # apenas caixas (classe x y largura altura)
                partes[0] = str(mapa[int(partes[0])])
                linhas.append(" ".join(partes))
    shutil.copy(imagens[0], pasta_img)
    with open(os.path.join(pasta_lbl, base + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + ("\n" if linhas else ""))
    n_imagens += 1
    n_objetos += len(linhas)

config = {"path": os.path.abspath(args.saida), "train": "images", "val": "images",
          "names": {i: n for i, n in enumerate(COCO)}}
with open(os.path.join(args.saida, "data.yaml"), "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

print(f"\n{n_imagens} imagens e {n_objetos} objetos anotados salvos em '{args.saida}'.")
print(f"Próximo passo: python avaliar_precisao.py --dados {args.saida}/data.yaml")
