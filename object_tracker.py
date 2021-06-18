import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image




def main(img, encoder, tracker, yolo, class_names, camera_id, num_classes = 80, resize = 416 ):
    
    fps = 0
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, resize)

    t1 = time.time()
    boxes, scores, classes, nums = yolo.predict(img_in)
    classes = classes[0]
    names = []
    for i in range(len(classes)):

        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)    
    detections = [Detection(bbox, score, class_name, feature) 
                  for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features) if class_name == 'person']
    
    #initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # # run non-maxima suppresion
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections] )
    indices = preprocessing.non_max_suppression(boxs, classes, 1, scores)
    detections = [detections[i] for i in indices]
    
    features = [i.feature for i in detections]
    
    
    
    # Call the tracker
    tracker.predict()
    tracker.update(detections)
    

    for track in tracker.tracks:

        if not track.is_confirmed() or track.time_since_update > 1:

            continue
    

        bbox = track.to_tlbr()
        class_name = track.get_class()

        
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
    # UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
    # for det in detections:
    #     bbox = det.to_tlbr() 
    #     cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
    
    # print fps on screen 
    # fps  = ( fps + (1./(time.time()-t1)) ) / 2
    # cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    # if camera_id == 0 :
    #     cv2.imshow(f'output{camera_id}', img)
    # else:
    #     cv2.imshow(f'output{camera_id}', img)
    
    # if cv2.waitKey(1) == ord('q'):
    #     cv2.destroyAllWindows()
    #     return

    return(tracker, img, features)
        

        
        



