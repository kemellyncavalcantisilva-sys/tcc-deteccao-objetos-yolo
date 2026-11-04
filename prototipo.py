import cv2
from ultralytics import YOLO
import time

# Carrega o modelo YOLO (versão leve e rápida)
model = YOLO("yolov8n.pt")

# Abre a webcam (0 = câmera padrão)
cap = cv2.VideoCapture(0)

# Variável para cálculo de FPS
prev_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao acessar a câmera")
        break

    # Faz a detecção
    results = model(frame)

    # Desenha os resultados no frame
    annotated_frame = results[0].plot()

    # Cálculo de FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Exibe o FPS na tela
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Mostra o vídeo
    cv2.imshow("Detecção de Objetos - YOLO", annotated_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera a câmera
cap.release()
cv2.destroyAllWindows()
