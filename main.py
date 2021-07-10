"""
Mobilenet is used only because it's high speed and
accuracy. It can also be used with Raspberry PI.
YOLO is not used since it requires GPU & YOLO Lite
is very tiny.

We can also we make our object detection trained model, but for beginners recommended is pre-trained model.
"""
# NMS = Non-Maximum Suppression
import cv2

# Adding the image to be read
# img = cv2.im read('lena.png')
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 480)
cap.set(10, 150)

# Adding the coco.names module to classNames
classNames =[]
classFile = 'coco.names'
with open(classFile, 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

# The Configuration and Weights Module
configF = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsF = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsF, configF)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

"""
oIds = Objects Class ID
bound = coordinates of the bounding box of Object
"""
while True:
    success, img = cap.read()
    oIds, confs, bound = net.detect(img, confThreshold=0.5)



    if len(oIds) != 0:
        for oId, confi, box in zip(oIds.flatten(), confs.flatten(), bound):
            cv2.rectangle(img, box, color=(208, 206, 64), thickness=3)
            cv2.putText(img, classNames[oId-1].upper(), (box[0]+10, box[1]+30),
                cv2.FONT_ITALIC, 1, (208, 206, 64), 1)
            cv2.putText(img, str(round(confi*100)), (box[0] + 10, box[1] + 60),
                        cv2.FONT_ITALIC, 1, (208, 206, 64), 1)
  





    cv2.imshow("Object Detected", img)

    cv2.waitKey(1)