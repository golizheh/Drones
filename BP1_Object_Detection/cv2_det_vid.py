import cv2
import utils
import argparse
import numpy as np
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config',
                help='path to yolo config file', default='ext/dr4e_5.cfg')
ap.add_argument('-w', '--weights',
                help='path to yolo pre-trained weights', default='ext/dr4e_5.weights')
ap.add_argument('-cl', '--classes',
                help='path to text file containing class names', default='ext/classes_5.names')
args = ap.parse_args()

# Define a window to show the cam stream on it
window_title = "Detector"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

# Extraction variables
number_faults = 0
faults_between_saving = 5
os.makedirs('./faults', exist_ok=True)

# Use this to define the faults the model must crop. 0 indicates how many faults it has found and is used for
# incrementing the file names
faults_list = {
    'vibration dampener': 0
}

# Load names classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

# Define network from configuration file and load the weights from the given weights file
net = cv2.dnn.readNet(args.weights, args.config)

# Define video capture for default cam, uncomment this line if you to use camera #instead of video
# cap = cv2.VideoCapture(0)
starting_time = time.time()
frame_id = 0
cap = cv2.VideoCapture("videos/power.webm")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)
# cap.set(3, 1280)
# cap.set(4, 720)
while cv2.waitKey(1) < 0:

    hasframe, image = cap.read()
    frame_id += 1
    # image=cv2.resize(image, (620, 480))
    # Extract blobs form images
    blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (320, 320), [0, 0, 0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)

    outs = net.forward(utils.getOutputsNames(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        # print(out.shape)
        for detection in out:

            # in YOLO each detection is in the form of "[center_x center_y width height obj_score class_1_score class_2_score ..]"
            scores = detection[5:]  # classes scores starts from index 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                ''' If we find a fault label from our fault list, we will crop the frame where we would normally put 
                the real time label box. We will then save the cropped image in a prespecified folder. I have noticed 
                the box coordinates are sometimes negative, so I make sure they don't go below 0.
                In order to not save too many pictures of the same fault, I have limited the problem to only save
                pictures every sixth detected fault. '''
                for fault in faults_list:
                    if classes[class_id] == fault:
                        if number_faults % faults_between_saving == 0:
                            number_faults += 1
                            pic_path = ('faults/{}_%s.jpg' % faults_list[fault]).format(fault)
                            faults_list[fault] += 1
                            crop_image = image[max(0, int(y)): max(0, int(y + h)), max(0, int(x)): max(0, int(x + w))]
                            cv2.imwrite(pic_path, crop_image)
                        else:
                            number_faults += 1

    # apply  non-maximum suppression algorithm on the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        utils.draw_pred(image, classes, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    # put frame/s
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(image, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    # Put efficiency information.
    t, _ = net.getPerfProfile()
    cv2.imshow(window_title, image)
