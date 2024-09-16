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


def main(dataset, nepochs, testArgs):
    # Carrega um dos modelos pré-treinados da ultralytics.
    # O modelo abaixo (yolov8s.pt) é o segundo em desempenho
    model = YOLO('yolov8s.pt')

    # executa o processamento na GPU
    model.to('cuda')

    # Treina o modelo para o dataset customizado com um número determinados de épocas. Quanto maior o número de
    # épocas, maior será a precisão do modelo resultante. Entretanto, o tempo para se realizar o treinamento pode ser
    # proibitivo. O treinamento termina quando o número de épocas for atingido, ou antes, quando os pesos deixam de
    # apresentar diferenças significativas entre diferentes execuções de épocas.
    model.train(data=dataset, epochs=nepochs)

    # Testa a performance do modelo usando o dataset de validação
    model.val()

    # Ao final do treinamento, o resultado do treinamento pode ser encontrado na pasta run. Procure pelo arquivo
    # "best.pt". Este arquivo contém os pesos do modelo customizado e pode ser usado para futuras detecções.

    # caso um arquivo de video tenha sido fornecido, realiza a detecção do objeto para o qual foi realizado o
    # treinamento
    if testArgs:
    
        cn = testArgs[0]
        videoForTest = testArgs[1]

        # carrega o video
        videoCap = cv2.VideoCapture(videoForTest)

        while True:

            ret, frame = videoCap.read()

            if not ret:
                continue

            # executa uma detecção usando o modelo yolov8 treinado anteriormente
            results = model.predict(frame, stream=True)


            for result in results:
                # pega os nomes de classes de objetos que foram achados no frame
                classes_names = result.names

                box_number = 0

                # para cada bounding box encontrada no resltado, verifica a qual classe de objeto ela pertente e
                # e desenha a bounding box em cima do objeto. Cada classe de objeto terá usa bounding box desenhada
                # com uma cor específica
                for box in result.boxes:

                    # obtem a classe do objeto detectado
                    cls = int(box.cls[0])

                    # obtem o nome da classe do objeto detectado
                    class_name = classes_names[cls]

                    # somente objetos com confidência acima de 80% e cuja classe seja a especificada pelo usuário terão
                    # bounding box desenhada na tela
                    if box.conf[0] > 0.8 and class_name == cn:
                        # pegas as coordenadas
                        [x1, y1, x2, y2] = box.xyxy[0]
                        # converte para int
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # pega a cor da classe
                        colour = getColours(cls)

                        # desenha a bounding box no objeto detectado
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                        # desenha o nome da classe e a confidência do objeto detectado
                        cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                        cv2.putText(frame, f'{str(box_number)}', (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                        box_number += 1


            # exibe a imagem
            cv2.imshow('frame', frame)

            # se a tecla 'q' for pressionada, para o processamento do video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # libera os recursos e fecha
        videoCap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinamento Yolo v8 com dataset customizado')
    parser.add_argument('dataset', default=None, help='caminho para o arquivo .yaml para o dataset customizado. O dataset deve estar localizado na pasta datasets')
    parser.add_argument('-e', '--epochs', default=500, help='número de épocas a serem usadas no treinamento', type=int)
    parser.add_argument('-t', '--test', nargs=2, metavar=('CLASS_NAME', 'VIDEO'), default=None, help='indica que depois de realizado o treinamento, uma etapda de testes de detecção será executada. Como parâmetros devem ser indicados a classe do objeto a ser detectado e o video onde será realizada a detecção')

    
    args = parser.parse_args()
    main(args.dataset, args.epochs, args.test)
