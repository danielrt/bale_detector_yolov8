import cv2
from ultralytics import YOLO
import argparse

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# obtem o centro de um fardo
def calCenter(bale):
    p1 = bale[1]
    p2 = bale[2]
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1])) / 2)

# verifica se dois fardos fazem parte da mesma pilha de fardos
def isStacked(bale1, bale2):
    c1 = calCenter(bale1)
    c2 = calCenter(bale2)
    if c1[0] == c2[0]:
        return True
    x1 = min(bale1[1][0], bale1[2][0])
    x2 = max(bale1[1][0], bale1[2][0])
    if x1 <= c2[0] <= x2:
        return True
    return False

# data um lista de fardos, retorna a relação entre eles: as pilhas de fardos existentes e quais fardos fazem parte de
# cada pilha. Considerando que o video de entrada possui duas pilhas de fardos, onde cada pilha possui dois fardos, está
# função iria gerar uma saida similar a abaixo:
# stacks:
# [
#   [
#       [3, (398, 170), (443, 207)],
#       [0, (399, 134), (444, 175)]
#   ],
#   [
#       [2, (459, 167), (504, 204)],
#       [1, (461, 128), (508, 173)]
#   ]
# ]

def detectBaleStacks(bales):
    bales.sort()
    stacks = []
    while len(bales):
        stack = []
        refBale = bales[0]
        stack.append(bales[0])
        bales.pop(0)
        i = 0
        while i < len(bales):
            if isStacked(refBale, bales[i]):
                stack.append(bales[i])
                bales.pop(i)
            else:
                i += 1
                
        stack.sort(key=lambda bale : calCenter(bale)[1], reverse=True)

        stacks.append(stack)

    return stacks


def main(video):

    # Carrega um modelo do yolo v8 que foi treinado com um dataset customizado para detectar os fardos.
    # O modelo "hercules500.pt" foi gerado através do scrip "yolo8_trainnig.py" com um dataset composto por 28 imagens.
    # O dataset foi criado pela ferramente de anotações roboflow (https://roboflow.com/)
    model = YOLO("hercules500.pt", verbose=False)

    # configura o yolo para executar na GPU
    model.to('cuda')

    # carrega o video onde devem ser detectadas as pilhas de fardos
    videoCap = cv2.VideoCapture(video)

    while True:

        ret, frame = videoCap.read()

        if not ret:
            continue

        # realiza a detecção de fardos em um frame
        results = model.predict(frame, stream=True)

        bales = []
        for result in results:
            # pega os nomes de classes de objetos que foram achados no frame
            classes_names = result.names

            box_number = 0

            # para cada bounding box encontrada no resltado, verifica se objeto detectado é um fardo
            # e desenha a bounding box em cima do objeto.
            for box in result.boxes:

                # obtem a classe do objeto detectado
                cls = int(box.cls[0])

                # obtem o nome da classe do objeto detectado
                class_name = classes_names[cls]

                # somente objetos do tipo fardo (class_name = bale) e com confidência acima de 80% são considerados
                if box.conf[0] > 0.8 and class_name == "bale":
                    # pegas as coordenadas
                    [x1, y1, x2, y2] = box.xyxy[0]
                    # converte para int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # adiona o fardo na lista de fardos. Cada fardo possui um número que o identifica
                    bales.append([box_number, (x1, y1), (x2, y2)])

                    # pega a cor da classe
                    colour = getColours(cls)

                    # desenha a bounding box no fardo detectado
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                    # desenha o nome da classe e a confidência do objeto detectado
                    cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                    cv2.putText(frame, f'{str(box_number)}', (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                    box_number += 1

        # caso algum fardo tenha sido detectado, realiza a detecção das pilhas de fardos e imprime o resultado no console
        if len(bales):
            print("stacks:")
            print(detectBaleStacks(bales))

        # exibe a imagem
        cv2.imshow('frame', frame)

        # se a tecla 'q' for pressionada, para o processamento do video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # libera os recursos e fecha
    videoCap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detecção de pilhas de fardos usando o Yolo v8')
    parser.add_argument('video', default=None, help='arquivo de video no qual será feita a detecção')

    args = parser.parse_args()
    main(args.video)
