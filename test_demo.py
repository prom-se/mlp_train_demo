import sys
import cv2
import os
import numpy as np

classes = open('model/classes.txt').read().splitlines()
net = cv2.dnn.readNetFromONNX('model/best_model.onnx')
img_path = 'data/MNIST/processed/mnist_test'
files = sorted(os.listdir(img_path))


def inference(input_image):
    cv2.imshow('img', input_image)
    blob = cv2.dnn.blobFromImage(input_image.astype(np.float32))
    net.setInput(blob)
    yhat = net.forward()
    index = np.argmax(yhat)
    conf = yhat[0][index]
    rst = classes[index]
    print(f'{img}:/TrueNumber:{file}/Predict:{rst}/Confidence:{conf:.2f}\r')
    return rst, conf


while True:
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    for file in files:
        images = sorted(os.listdir(os.path.join(img_path, file)))
        for img in images:
            src = cv2.imread(os.path.join(img_path, file, img), cv2.IMREAD_GRAYSCALE) / 255.0
            inference(src)
            if cv2.waitKey(250) & 0xFF == ord('r'):
                break
            if cv2.waitKey(250) & 0xFF == ord('q'):
                sys.exit(0)
