# Detecção de Pilhas de Fardos Usando o Yolo8

Este repositório contém os códigos dos experimentos realizados para a detecção de fardos / pilhas de fardos no contexto do projeto do robo Hércules.

O processo foi constituído por duas etapas:

1. Treinamento de um dos modelos do Yolo v8 para detectar pilhas de fardos.
	- Foram capturadas várias imagens dos blocos/fardos, em diferentes posições e orientações. Essas imagens se encontram na pasta images.
	- As imagens capturadas foram anotadas com a ferramenta web [Roboflow](https://roboflow.com/).
	- Com o dataset de treinamento gerado pelo Roboflow
	
2. Detecção das pilhas de fardos usando o modelo treinado na etapa 1.



## Dependências
Para utilizar os códigos execute as linhas de comando a seguir para instalar as bibliotecas Python necessárias:

```commandline
pip install torch torchvision torchaudio
pip install opencv-python
pip install ultralytics
```

## Treinamento da Yolo v8 para detectar um objeto específico

Esta seção contém informações sobre como foi feito o processo de treinamento para que a Yolo v8 fosse capaz de detectar 
as pilhas de fardos. Apesar de ter sido realizado para um caso específico, o processo utilizado pode ser realizado para 
qualquer tipo de objeto. Caso você queira apenas executar a detecção das pilhas de fardos, pule para a próxima seção.

Para realizar o treinamento do yolo para detectar um objeto específico, foram realizadas as seguintes etapas:

1. Captura de várias imagens do objeto, em posições e orientações diferentes.
	- para o caso específico de detecção dos blocos/fardos, foram capturadas 28 imagens. Essas se encontram na pasta "images".
2. Criação do dataset customizado de treinamento, com a anotação das imagens capturadas na etapa 1.
	- Para anotar as imagens foi utilizada a ferramente web [Roboflow](https://roboflow.com/). O link 
	(https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/#preparing-a-custom-dataset-for-yolov8) mostra em detalhes 
	como a ferramenta pode ser usada para criar classes de objetos, anotar as imagens e exportar um dataset que será usado para treinar 
	o yolo8.
3. Treinamento da rede yolo8 com o dataset criado na etapa anterior usando o script "yolo8_trainning.py".
	- Para o caso específico de detecão de fardos, o Roboflow gerou o arquivo "hercules.v1i.yolov8.zip". Esse arquivo foi descompactado 
	na pasta "datasets", sem nenhuma pasta intermediária. **ATENÇÃO**: o yolo exige que os datasets customizados estejam na pasta "datasets".
	- O script "yolo8_trainning.py" aceita os seguintes parâmetros:
	
```
usage: yolo8_trainning.py [-h] [-e EPOCHS] [-t CLASS_NAME VIDEO] dataset

Treinamento Yolo v8 com dataset customizado

positional arguments:
  dataset               caminho para o arquivo .yaml para o dataset customizado. O dataset deve estar localizado na pasta datasets

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        número de épocas a serem usadas no treinamento
  -t CLASS_NAME VIDEO, --test CLASS_NAME VIDEO
                        indica que depois de realizado o treinamento, uma etapda de testes de detecção será executada. Como parâmetros devem ser indicados a classe do objeto a ser detectado e o video
                        onde será realizada a detecção

Exemplo de uso do script de treinamento:

python3 yolo8_trainning.py datasets/data.yaml --epochs=5 --test bale videos/5.mp4

```

4. Ao terminar a execução do treinamento, o arquivo com os pesos otimizados para detecção de objeto específico usando o Yolo estará em uma das subpastas da pasta "runs", sob o nome **"best.pt"**
	- Para o caso específico de detecão de fardos, o treinamento foi feito com o parâmetro de 500 épocas e gerou um arquivo de pesos. Esse arquivo foi renomeado para "hercules500.pt" e se encontra na raiz do repositório. Ele é utilizado pelo processo de detecção descrito na próxima seção. 
	
**Não há necessidade de regerar o modelo se apenas se quer realizar o processo de detecção**.

## Detecção das pilhas de fardos usando o Yolo v8

Para o processo de detecção das pilhas de fardos, você pode usar diretamente o script "bale_yolo8_detection.py", sem a necessidade de realizar o treinamento do Yolo. O script usa o arquivo "hercules500.pt" como input para o Yolo. Esse arquivo foi gerado através da execução do treinamento para o caso específico para detecção de fardos (ver seção anterior). Para usar o script:

```
usage: bale_yolo8_detection.py [-h] video

Detecção de pilhas de fardos usando o Yolo v8

positional arguments:
  video       arquivo de video no qual será feita a detecção

optional arguments:
  -h, --help  show this help message and exit

```

A pasta videos contém 5 videos que podem ser usados para testar o processo de detecção. Abaixo um exemplo de uso do script:

```
python3 bale_yolo8_detection.py videos/5.mp4 

```

Como resultado, você deveria ver um video com os fardos sendo marcados pelo desenho das caixas envolventes. No console, uma sequência de mensagens do tipo:

```
0: 384x640 4 bales, 12.7ms
Speed: 3.4ms preprocess, 12.7ms inference, 2.2ms postprocess per image at shape (1, 3, 384, 640)
stacks:
[
	[
		[3, (398, 170), (443, 207)], 
		[0, (399, 134), (444, 175)]
	], 
	[
		[2, (459, 167), (504, 204)],
		[1, (461, 128), (508, 173)]
	]
]


```
A saída acima indica que foram detectados 4 fardos, sendo que eles estão dispostos em duas pilhas. A primera contém os fardos 3 e 0, enquanto que a segunda contém os fardos 2 e 1.

