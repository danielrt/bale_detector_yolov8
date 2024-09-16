import cv2
from ultralytics import YOLO


# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)


def calCenter(bale):
    p1 = bale[1]
    p2 = bale[2]
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1])) / 2)


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


def main():
    # Load the model
    #yolo = YOLO('yolov8s.pt')

    # Create a new YOLO model from scratch
    #model = YOLO("yolov8n.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("best500.pt", verbose=False)
    model.to('cuda')

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    #results = model.train(data="hercules.v1i.yolov8/data.yaml", epochs=500)

    # Evaluate the model's performance on the validation set
    #results = model.val()

    # Load the video capture
    videoCap = cv2.VideoCapture("5.mp4")

    while True:

        ret, frame = videoCap.read()
        #frame = cv2.imread('IMG_20240730_144739032.jpg')
        if not ret:
            continue
        results = model.predict(frame, stream=True)

        bales = []
        for result in results:
            # get the classes names
            classes_names = result.names

            box_number = 0

            # iterate over each box
            for box in result.boxes:

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # check if confidence is greater than 80 percent
                if box.conf[0] > 0.8 and class_name == "bale":
                    # get coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]
                    # convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    bales.append([box_number, (x1, y1), (x2, y2)])

                    # get the respective colour
                    colour = getColours(cls)

                    # draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                    # put the class name and confidence on the image
                    cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                    cv2.putText(frame, f'{str(box_number)}', (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                    box_number += 1

        if len(bales):
            print("stacks:")
            print(detectBaleStacks(bales))

        # show the image
        cv2.imshow('frame', frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the video capture and destroy all windows
    videoCap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
