"""
detector.py - Sistema de detecção de objetos em tempo real (versão final)
Autora: Kemellyn Cavalcanti da Silva

Aplicação para o usuário final: abre a webcam, detecta objetos com um modelo
YOLO e mostra na tela as caixas, o FPS e a contagem de objetos por classe.

Uso:
  python detector.py
  python detector.py --camera 1 --modelo yolo11n.pt

Teclas durante a execução:
  1, 2, 3, 4  trocam o modelo (YOLOv8n, YOLOv8s, YOLO11n, YOLO11s)
  S           salva a imagem atual na pasta "capturas"
  Q ou ESC    encerra o programa
"""

import argparse
import os
import time
from collections import Counter, deque
from datetime import datetime

import cv2
from ultralytics import YOLO

MODELOS = {ord("1"): "yolov8n.pt", ord("2"): "yolov8s.pt",
           ord("3"): "yolo11n.pt", ord("4"): "yolo11s.pt"}
JANELA = "Deteccao de Objetos em Tempo Real - YOLO"  # sem acentos (limitação do Windows)
PASTA_CAPTURAS = "capturas"


def abrir_camera(indice):
    """Tenta abrir a câmera pelo DirectShow e, se falhar, pelo modo padrão."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(indice, backend)
        if cap.isOpened():
            for _ in range(30):
                ok, _frame = cap.read()
                if ok:
                    return cap
                time.sleep(0.1)
        cap.release()
    return None


def localizar_camera(preferida):
    """Usa a câmera indicada; se ela não responder, procura outra (0 a 3)."""
    for indice in [preferida] + [i for i in range(4) if i != preferida]:
        cap = abrir_camera(indice)
        if cap is not None:
            if indice != preferida:
                print(f"Câmera {preferida} indisponível. Usando a câmera {indice}.")
            return cap, indice
    return None, None


def carregar_modelo(nome):
    print(f"Carregando o modelo {nome}...")
    return YOLO(nome)


def desenhar_painel(img, nome_modelo, fps, contagem):
    """Desenha o painel de informações no canto superior esquerdo."""
    linhas = [f"Modelo: {nome_modelo.replace('.pt', '')}",
              f"FPS: {fps:.1f}",
              f"Objetos: {sum(contagem.values())}"]
    linhas += [f"  {classe}: {qtd}" for classe, qtd in contagem.most_common(5)]

    altura = 30 + 26 * len(linhas)
    fundo = img.copy()
    cv2.rectangle(fundo, (10, 10), (290, altura), (0, 0, 0), -1)
    cv2.addWeighted(fundo, 0.55, img, 0.45, 0, img)
    for i, texto in enumerate(linhas):
        cv2.putText(img, texto, (20, 38 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if i < 3 else (255, 255, 255), 2 if i < 3 else 1)

    ajuda = "1-4: trocar modelo | S: salvar imagem | Q: sair"
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 30), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, ajuda, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    p = argparse.ArgumentParser(description="Detecção de objetos em tempo real com YOLO")
    p.add_argument("--camera", type=int, default=0, help="número da câmera (padrão: 0)")
    p.add_argument("--modelo", default="yolov8n.pt", help="modelo inicial (padrão: yolov8n.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="resolução de entrada do modelo")
    p.add_argument("--conf", type=float, default=0.25, help="confiança mínima das detecções")
    args = p.parse_args()

    cap, indice = localizar_camera(args.camera)
    if cap is None:
        raise SystemExit("Nenhuma câmera encontrada. Verifique a conexão da webcam, se outro "
                         "programa a está usando e as permissões em Configurações > "
                         "Privacidade e segurança > Câmera.")

    nome_modelo = args.modelo
    modelo = carregar_modelo(nome_modelo)
    tempos = deque(maxlen=30)  # média móvel dos últimos 30 quadros: FPS estável na tela
    falhas = 0
    print(f"Sistema iniciado na câmera {indice}. Pressione Q na janela para sair.")

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            falhas += 1
            if falhas > 30:
                print("A câmera parou de responder.")
                break
            continue
        falhas = 0

        r = modelo(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        img = r.plot()
        contagem = Counter(r.names[int(c)] for c in r.boxes.cls)

        tempos.append(time.perf_counter() - t0)
        fps = len(tempos) / sum(tempos)
        desenhar_painel(img, nome_modelo, fps, contagem)
        cv2.imshow(JANELA, img)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27):
            break
        if tecla in (ord("s"), ord("S")):
            os.makedirs(PASTA_CAPTURAS, exist_ok=True)
            arquivo = os.path.join(PASTA_CAPTURAS, datetime.now().strftime("captura_%Y%m%d_%H%M%S.jpg"))
            cv2.imwrite(arquivo, img)
            print(f"Imagem salva em {arquivo}")
        if tecla in MODELOS and MODELOS[tecla] != nome_modelo:
            aviso = img.copy()
            cv2.putText(aviso, "Carregando modelo...", (20, img.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imshow(JANELA, aviso)
            cv2.waitKey(1)
            nome_modelo = MODELOS[tecla]
            modelo = carregar_modelo(nome_modelo)
            tempos.clear()
        # Encerra também se o usuário fechar a janela pelo "X"
        if cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
