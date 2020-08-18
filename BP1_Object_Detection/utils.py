import cv2
import random
import numpy as np

# Get names of output layers, output for YOLO
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(80, 3))
# Darw a rectangle surrounding the object and its class name
def draw_pred(img, classes, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
