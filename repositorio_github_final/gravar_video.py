"""
gravar_video.py - Grava um vídeo de teste com a webcam (TCC)

Todos os modelos serão avaliados sobre o MESMO vídeo gravado, garantindo uma
comparação justa e reproduzível (com a webcam ao vivo, a cena muda a cada teste).

Uso:
  python gravar_video.py videos/normal.mp4 --segundos 60
  python gravar_video.py videos/pouca_luz.mp4 --segundos 60
Pressione 'q' para encerrar antes do tempo.
"""

import argparse
import os
import time

import cv2

p = argparse.ArgumentParser(description="Gravação de vídeo de teste")
p.add_argument("saida", help="caminho do arquivo .mp4 a ser criado")
p.add_argument("--segundos", type=int, default=60)
p.add_argument("--camera", type=int, default=0)
p.add_argument("--largura", type=int, default=640)
p.add_argument("--altura", type=int, default=480)
args = p.parse_args()

os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
# DirectShow primeiro: no Windows, o modo padrão (MSMF) pode falhar nos primeiros quadros
cap, ok, frame = None, False, None
for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.altura)
    for _ in range(50):
        ok, frame = cap.read()
        if ok:
            break
        time.sleep(0.1)
    if ok:
        break
    cap.release()
if not ok:
    raise SystemExit("Não foi possível acessar a câmera. Verifique se outro programa está "
                     "usando a câmera e as permissões em Configurações > Privacidade > Câmera.")
h, w = frame.shape[:2]

# Mede o FPS real da câmera para o vídeo não ficar acelerado ou lento
t0 = time.perf_counter()
for _ in range(30):
    cap.read()
fps_real = 30 / (time.perf_counter() - t0)

gravador = cv2.VideoWriter(args.saida, cv2.VideoWriter_fourcc(*"mp4v"), fps_real, (w, h))
inicio = time.perf_counter()
n = 0
while time.perf_counter() - inicio < args.segundos:
    ok, frame = cap.read()
    if not ok:
        break
    gravador.write(frame)
    n += 1
    previa = frame.copy()
    restante = args.segundos - (time.perf_counter() - inicio)
    cv2.putText(previa, f"REC  {restante:.0f}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Gravacao - pressione q para parar", previa)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
gravador.release()
cv2.destroyAllWindows()
print(f"Vídeo salvo em {args.saida}: {n} quadros, {w}x{h}, ~{fps_real:.1f} FPS da câmera.")
