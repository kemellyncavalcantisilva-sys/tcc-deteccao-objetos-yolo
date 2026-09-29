"""
listar_cameras.py - Descobre o número de cada câmera conectada ao computador.

Uso:
  python listar_cameras.py

Cada câmera encontrada aparece numa janela por alguns segundos, com o número dela
escrito na imagem. Use esse número no parâmetro --fonte dos outros scripts.
"""

import time

import cv2

print("Procurando câmeras (0 a 4)...\n")
encontradas = []

for indice in range(5):
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(30):
            ok, frame = cap.read()
            if ok:
                break
            time.sleep(0.1)
    if not ok:
        print(f"Câmera {indice}: não encontrada")
        cap.release()
        continue

    h, w = frame.shape[:2]
    preta = frame.mean() < 10
    situacao = "IMAGEM PRETA (bloqueada ou em uso)" if preta else "funcionando"
    print(f"Câmera {indice}: {w}x{h} - {situacao}")
    encontradas.append(indice)

    inicio = time.time()
    while time.time() - inicio < 5:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"CAMERA {indice}  (use --fonte {indice})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Teste de cameras - aguarde", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

print(f"\nCâmeras com sinal: {encontradas if encontradas else 'nenhuma'}")
