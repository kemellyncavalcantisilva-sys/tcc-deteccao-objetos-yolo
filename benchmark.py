"""
benchmark.py - Avaliação de desempenho de modelos YOLO (TCC)
Autora: Kemellyn Cavalcanti 

Mede, para cada modelo, a velocidade de processamento (FPS e latência) e o
comportamento das detecções sobre a MESMA fonte de vídeo, salvando tudo em CSV
(formato brasileiro: separador ";" e decimal ",", abre direto no Excel).

Exemplos de uso:
  python benchmark.py --fonte videos/normal.mp4 --condicao normal --repeticoes 3
  python benchmark.py --fonte videos/pouca_luz.mp4 --condicao pouca_luz --repeticoes 3
  python benchmark.py --fonte 0 --condicao ao_vivo --quadros 300 --mostrar
  python benchmark.py --modelos yolov8n.pt yolov8n.onnx --fonte videos/normal.mp4 --condicao otimizacao
  python benchmark.py --modelos yolov8n.pt --imgsz 320 --fonte videos/normal.mp4 --condicao imgsz_320
"""

import argparse
import csv
import os
import statistics
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

MODELOS_PADRAO = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"]
PASTA_RESULTADOS = "resultados"
# Sem acentos: o cv2.imshow no Windows não exibe acentos no título da janela
TITULO_JANELA = "Deteccao de Objetos - YOLO"

CABECALHO_QUADROS = [
    "data_hora", "modelo", "condicao", "rodada", "imgsz", "quadro",
    "latencia_ms", "pre_ms", "inferencia_ms", "pos_ms",
    "deteccoes", "confianca_media", "classes",
]


def ler_argumentos():
    p = argparse.ArgumentParser(description="Benchmark de modelos YOLO")
    p.add_argument("--modelos", nargs="+", default=MODELOS_PADRAO,
                   help="modelos a comparar (.pt, .onnx ou pasta OpenVINO)")
    p.add_argument("--fonte", default="0", help="0 para webcam ou caminho de um vídeo")
    p.add_argument("--condicao", default="padrao",
                   help="rótulo do cenário de teste (ex.: normal, pouca_luz, movimento)")
    p.add_argument("--quadros", type=int, default=500, help="máximo de quadros medidos por rodada")
    p.add_argument("--aquecimento", type=int, default=10,
                   help="inferências descartadas antes da medição")
    p.add_argument("--repeticoes", type=int, default=1, help="rodadas por modelo")
    p.add_argument("--imgsz", type=int, default=640, help="resolução de entrada do modelo")
    p.add_argument("--conf", type=float, default=0.25, help="confiança mínima das detecções")
    p.add_argument("--dispositivo", default="cpu", help="cpu (notebook sem GPU NVIDIA)")
    p.add_argument("--mostrar", action="store_true",
                   help="exibe o vídeo anotado (útil para prints; reduz um pouco o FPS)")
    return p.parse_args()


def num(valor, casas=2):
    """Formata número com vírgula decimal para o Excel em português."""
    return f"{valor:.{casas}f}".replace(".", ",")


def percentil(valores, p):
    ordenados = sorted(valores)
    k = int(round((p / 100) * (len(ordenados) - 1)))
    return ordenados[k]


def abrir_fonte(fonte):
    """Abre a webcam ou um arquivo de vídeo.

    Na webcam, usa o DirectShow (CAP_DSHOW) primeiro: no Windows, o modo padrão
    (MSMF) pode falhar na leitura dos primeiros quadros. Se o DirectShow não
    funcionar, tenta o modo padrão. A câmera só é aceita depois de entregar um
    quadro válido.
    """
    if not fonte.isdigit():
        cap = cv2.VideoCapture(fonte)
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {fonte}")
        return cap

    indice = int(fonte)
    for backend, nome in ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "padrão")):
        cap = cv2.VideoCapture(indice, backend)
        if cap.isOpened():
            for _ in range(50):  # a câmera pode levar alguns instantes para "acordar"
                ok, _frame = cap.read()
                if ok:
                    print(f"Câmera {indice} aberta (modo {nome}).")
                    return cap
                time.sleep(0.1)
        cap.release()
    raise RuntimeError(
        f"Não foi possível ler a câmera {fonte}. Verifique se outro programa está usando a "
        "câmera e se o acesso está liberado em Configurações > Privacidade e segurança > Câmera."
    )


def info_modelo(modelo, nome):
    try:
        parametros = sum(p.numel() for p in modelo.model.parameters())
    except Exception:
        parametros = ""  # modelos exportados (ONNX/OpenVINO) não expõem os parâmetros
    tamanho_mb = num(os.path.getsize(nome) / 1e6) if os.path.isfile(nome) else ""
    return parametros, tamanho_mb


def inferir(modelo, frame, args):
    return modelo(frame, imgsz=args.imgsz, conf=args.conf,
                  device=args.dispositivo, verbose=False)[0]


def aquecer(modelo, args):
    """Executa inferências iniciais descartadas: as primeiras são sempre mais lentas."""
    cap = abrir_fonte(args.fonte)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Não foi possível ler o primeiro quadro para o aquecimento.")
    for _ in range(args.aquecimento):
        inferir(modelo, frame, args)


def medir(modelo, nome, rodada, data_hora, args, escritor):
    cap = abrir_fonte(args.fonte)  # reabre: todos os modelos processam os mesmos quadros
    m = {"lat": [], "pre": [], "inf": [], "pos": [], "det": [], "conf": []}
    classes_vistas = {}

    while len(m["lat"]) < args.quadros:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        r = inferir(modelo, frame, args)
        ms = (time.perf_counter() - t0) * 1000  # ciclo: leitura do quadro + detecção

        n = len(r.boxes)
        conf_media = float(r.boxes.conf.mean()) if n else 0.0
        nomes = [r.names[int(c)] for c in r.boxes.cls]
        for c in nomes:
            classes_vistas[c] = classes_vistas.get(c, 0) + 1

        m["lat"].append(ms)
        m["pre"].append(r.speed["preprocess"])
        m["inf"].append(r.speed["inference"])
        m["pos"].append(r.speed["postprocess"])
        m["det"].append(n)
        if n:
            m["conf"].append(conf_media)

        escritor.writerow([
            data_hora, nome, args.condicao, rodada, args.imgsz, len(m["lat"]),
            num(ms), num(r.speed["preprocess"]), num(r.speed["inference"]),
            num(r.speed["postprocess"]), n, num(conf_media, 3), "|".join(nomes),
        ])

        if args.mostrar:
            img = r.plot()
            cv2.putText(img, f"{nome}  FPS: {1000 / ms:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow(TITULO_JANELA, img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    return m, classes_vistas


def resumir(m, classes_vistas, nome, rodada, data_hora, parametros, tamanho_mb, args):
    lat = m["lat"]
    fps_quadro = [1000 / x for x in lat]
    total = len(lat)
    classes = sorted(classes_vistas.items(), key=lambda kv: -kv[1])
    return {
        "data_hora": data_hora,
        "modelo": nome,
        "condicao": args.condicao,
        "rodada": rodada,
        "imgsz": args.imgsz,
        "quadros": total,
        "fps_medio": num(total / (sum(lat) / 1000)),
        "fps_desvio": num(statistics.stdev(fps_quadro) if total > 1 else 0),
        "fps_min": num(min(fps_quadro)),
        "latencia_media_ms": num(statistics.mean(lat)),
        "latencia_p95_ms": num(percentil(lat, 95)),
        "pre_ms": num(statistics.mean(m["pre"])),
        "inferencia_ms": num(statistics.mean(m["inf"])),
        "pos_ms": num(statistics.mean(m["pos"])),
        "deteccoes_media": num(statistics.mean(m["det"])),
        "quadros_sem_deteccao_pct": num(100 * sum(1 for d in m["det"] if d == 0) / total),
        "confianca_media": num(statistics.mean(m["conf"]) if m["conf"] else 0, 3),
        "classes_detectadas": "; ".join(f"{k}: {v}" for k, v in classes),
        "parametros": parametros,
        "tamanho_mb": tamanho_mb,
    }


def salvar_resumo(resumo):
    caminho = os.path.join(PASTA_RESULTADOS, "resumo.csv")
    novo = not os.path.exists(caminho)
    with open(caminho, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(resumo.keys()), delimiter=";")
        if novo:
            w.writeheader()
        w.writerow(resumo)


def main():
    args = ler_argumentos()
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    caminho_quadros = os.path.join(PASTA_RESULTADOS, "quadros.csv")
    novo = not os.path.exists(caminho_quadros)

    with open(caminho_quadros, "a", newline="", encoding="utf-8-sig") as fq:
        escritor = csv.writer(fq, delimiter=";")
        if novo:
            escritor.writerow(CABECALHO_QUADROS)

        for nome in args.modelos:
            print(f"\n=== {nome} | condição: {args.condicao} | imgsz: {args.imgsz} ===")
            modelo = YOLO(nome)
            parametros, tamanho_mb = info_modelo(modelo, nome)
            aquecer(modelo, args)

            for rodada in range(1, args.repeticoes + 1):
                data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                m, classes = medir(modelo, nome, rodada, data_hora, args, escritor)
                if not m["lat"]:
                    print("Nenhum quadro processado.")
                    continue
                r = resumir(m, classes, nome, rodada, data_hora, parametros, tamanho_mb, args)
                salvar_resumo(r)
                print(f"Rodada {rodada}: FPS médio {r['fps_medio']} (±{r['fps_desvio']}) | "
                      f"inferência {r['inferencia_ms']} ms | "
                      f"detecções/quadro {r['deteccoes_media']} | confiança {r['confianca_media']}")

    cv2.destroyAllWindows()
    print(f"\nResultados salvos na pasta '{PASTA_RESULTADOS}'.")


if __name__ == "__main__":
    main()
