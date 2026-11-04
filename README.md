# Detecção de Objetos em Tempo Real com YOLO

Trabalho de Conclusão de Curso – Ciência da Computação, UNIP (2026)  
**Autora:** Kemellyn Cavalcanti da Silva · **Orientadora:** Prof.ª Me. Sandra Muniz Bozolan

Sistema de detecção de objetos em tempo real com webcam, desenvolvido em Python com OpenCV e Ultralytics, e conjunto de scripts para comparar o desempenho dos modelos **YOLOv8n, YOLOv8s, YOLO11n e YOLO11s** em um computador sem placa de vídeo dedicada.

## Principais resultados

| Modelo | FPS médio | Inferência (ms) | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| YOLOv8n | 18,9 | 45,3 | 0,431 | 0,368 | 0,445 | 0,162 |
| YOLOv8s | 8,7 | 107,7 | 0,487 | 0,471 | 0,511 | 0,202 |
| YOLO11n | 17,1 | 51,3 | 0,531 | 0,402 | 0,439 | 0,161 |
| YOLO11s | 8,6 | 108,6 | 0,668 | 0,517 | 0,533 | 0,195 |

Notebook AMD Ryzen 5 5500U, 8 GB de RAM, sem GPU NVIDIA. FPS = média de quatro condições de teste (normal, pouca luz, movimento e vários objetos), com três repetições cada. Precisão medida sobre 40 imagens anotadas manualmente. Os dados completos estão na pasta `resultados`.

## Arquivos

| Arquivo | Função |
|---|---|
| `detector.py` | Versão final do sistema de detecção em tempo real |
| `prototipo_inicial.py` | Primeira versão do sistema |
| `listar_cameras.py` | Mostra as câmeras disponíveis e o número de cada uma |
| `gravar_video.py` | Grava os vídeos de teste |
| `benchmark.py` | Executa os modelos sobre um vídeo e registra FPS, latência e detecções |
| `extrair_quadros.py` | Extrai quadros dos vídeos para anotação |
| `anotar.py` | Anotador manual: desenha as caixas e salva no formato YOLO com a numeração do COCO |
| `converter_rotulos.py` | Alternativa: converte anotações exportadas do Roboflow para a numeração do COCO |
| `avaliar_precisao.py` | Calcula Precision, Recall e mAP sobre o conjunto anotado |
| `comparar_quadro.py` | Gera uma imagem comparando as detecções dos modelos em um mesmo quadro |
| `requirements.txt` | Dependências do projeto |
| `resultados/` | Resultados dos experimentos (`resumo.csv` e `precisao.csv`) |

## Requisitos

- Windows, webcam e internet na primeira execução (os modelos são baixados automaticamente)
- [Python 3.12](https://www.python.org/downloads/release/python-31210/), instalado com a opção **"Add python.exe to PATH"** marcada

## Instalação

Abra o terminal na pasta do projeto e execute:

```
python -m pip install -r requirements.txt
```

## Uso do sistema

```
python detector.py
python detector.py --camera 1     # para escolher outra câmera
```

| Tecla | Função |
|---|---|
| `1` `2` `3` `4` | Trocar o modelo (YOLOv8n, YOLOv8s, YOLO11n, YOLO11s) |
| `S` | Salvar a imagem atual na pasta `capturas` |
| `Q` ou `ESC` | Encerrar |

O painel no canto superior esquerdo mostra o modelo em uso, o FPS (média dos últimos 30 quadros) e a contagem de objetos por classe.

## Reprodução dos experimentos

```
# 1. Descobrir o número da câmera
python listar_cameras.py

# 2. Gravar um vídeo de 20 segundos por condição de teste
python gravar_video.py videos/normal.mp4 --segundos 20 --camera 1
python gravar_video.py videos/pouca_luz.mp4 --segundos 20 --camera 1
python gravar_video.py videos/movimento.mp4 --segundos 20 --camera 1
python gravar_video.py videos/varios_objetos.mp4 --segundos 20 --camera 1

# 3. Medir a velocidade dos 4 modelos em cada vídeo (3 repetições)
python benchmark.py --fonte videos/normal.mp4 --condicao normal --repeticoes 3
python benchmark.py --fonte videos/pouca_luz.mp4 --condicao pouca_luz --repeticoes 3
python benchmark.py --fonte videos/movimento.mp4 --condicao movimento --repeticoes 3
python benchmark.py --fonte videos/varios_objetos.mp4 --condicao varios_objetos --repeticoes 3

# 4. Avaliar a precisão
python extrair_quadros.py --por-video 10
python anotar.py
python avaliar_precisao.py --dados dataset_coco/data.yaml

# 5. Comparar visualmente os modelos em um quadro
python comparar_quadro.py --video videos/pouca_luz.mp4 --saida figura_pouca_luz.jpg
```

Os arquivos CSV usam `;` como separador e vírgula decimal, e abrem direto no Excel em português.

Os vídeos de teste e as imagens anotadas não estão no repositório, porque mostram a autora e o ambiente onde foram gravados.

## Ambiente dos testes

Notebook Acer Aspire 5 (AMD Ryzen 5 5500U, 8 GB de RAM, sem GPU NVIDIA), Windows, Python 3.12.10, Ultralytics 8.4.165, PyTorch em CPU.
