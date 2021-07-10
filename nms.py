"""
Mobilenet is used only bcoz it's high speed and
accuracy. It can also be used with Raspberry PI.
YOLO is not used since it requires GPU & YOLO Lite
is very tiny.

We can also we make our object detection trained model, but for beginners recommended is pre-trained model.
"""
# NMS = Non-Maximum Suppression
import cv2
import numpy as np

# Adding the image to be read
# img = cv2.imread('lena.png')
thres = 0.5
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 480)
cap.set(10, 150)

# Adding the coco.names module to classNames
classNames = []
classFile = 'coco.names'
with open(classFile, 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

# The Configuration and Weights Module
configF = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsF = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsF, configF)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

"""
oIds = Objects Class ID
bound = coordinates of the bounding box of Object
"""
while True:
    success, img = cap.read()
    oIds, confs, bound = net.detect(img, confThreshold=thres)
    bound = list(bound)
    # Here we use numpy reason being tuples don't have reshape attribute
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bound, confs, thres, nms_threshold=0.35)

    for i in indices:
        i = i[0]
        box = bound[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(210, 205, 64), thickness=2)
        cv2.putText(img, classNames[oIds[i][0]-1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_ITALIC, 1, (208, 206, 64), 1)


    cv2.imshow("Object Detected", img)

    cv2.waitKey(1)