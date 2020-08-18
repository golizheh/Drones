# opencv implementation of yolo

This repo provides a clean implementation of YoloV3 and yolov3-tiny within opencv environment.

## Key Features

- [x] opencv (4.2.0)


## Usage
#### Pip

```bash
pip install opencv-python
```
## Downloads 
 ```
 #change directory to "ext" and run the following command to download weights.
cd ext
https://nextcloud.sdu.dk/index.php/s/4Hnej2aZ9ANedaQ

```

### Detection

```
python cv2_det_vid.py
if you want to use camera instead of video you can uncomment the line 30 
"cap = cv2.VideoCapture(0)"
and comment out the line (33)
"cap = cv2.VideoCapture("crowd.mp4")"
```

###  Args 
```
cv2_det_vid.py
    --config: path to config file
    (default: 'ext/yolov3.cfg')
    --weights: path to pre-trained weights file
    (default: 'ext/yolov3.weights')
    --classes: path to classes file
          (default: 'ext/classes.names')
```
##  To do 
Extract the detected objects from the video files in the form of images and store it in seprate folder



Note: Always remember to add the comments related to your updates in this code.


## References

 You can follow the official codes to understand the main functionality.

- https://github.com/pjreddie/darknet
    - official yolov3 implementation
- https://github.com/AlexeyAB
    - explinations of parameters
