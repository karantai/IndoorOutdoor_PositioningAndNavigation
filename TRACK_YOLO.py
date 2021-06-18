import cv2
import numpy as np
import create_3d_bb
# import solve_equations
# from  undistort_RATIONAL_without_otpimal import undistorted
# import time
# from  matching_bb import correlate_bb
# import threading
import object_tracker
# from scipy import spatial
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
import logging
import tensorflow as tf
from scipy import spatial
from sympy.geometry import *
import skgeom as sg
from change_detection_bb_YOLO import Detects



def triangulate(imgp4, imgp3, projL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/projection_matrixL.npy'), \
                projR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/projection_matrixR.npy')):
    test_points = cv2.triangulatePoints(projL, projR, np.float32(imgp4), np.float32(imgp3))
    # breakpoint()
    test_points = test_points[:3, :] / test_points[3, :]
    test_points = np.around(test_points, 3).T
    
    projected = np.int32(cv2.projectPoints(test_points, cv2.Rodrigues(
        np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/rotation_matrixR.npy'))[0], 
        np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/tvecs3.npy'), 
        np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_camera.npy'), np.zeros((1,5)))[0])
    return projected, test_points

          
if __name__=='__main__':
    

    
    names = ['right_cutt_video.mp4', 'left_cutt_video.mp4']
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = 'model_data/mars-small128.pb'
    
    encoderR = gdet.create_box_encoder(model_filename, batch_size=1)
    encoderL = gdet.create_box_encoder(model_filename, batch_size=1)
    
    classes_path = './data/labels/coco.names'
    weights_path = './weights/yolov3.tf'    
    
    metricL = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metricR = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    
    

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


    
    trackerL = Tracker(metricL)
    trackerR = Tracker(metricR)
   
    yoloR = YoloV3(classes = 80)
    yoloL = YoloV3(classes = 80)

    yoloR.load_weights(weights_path)
    yoloL.load_weights(weights_path)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes_path).readlines() ]
    logging.info('classes loaded')


    count = 0

    videoR = cv2.VideoCapture(names[0])
    videoL = cv2.VideoCapture(names[1])
    results = {}
    case4 = np.load('table_coords_left.npy')
    case3 = np.load('table_coords_right.npy')
    
    x, table_3d = triangulate(case4, case3)
    lines_table = []
    
    mtxR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_camera.npy')
    distR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_distortion.npy')
    projR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/projection_matrixR.npy')
    rot_mat3 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/rotation_matrixR.npy') 
    distR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/right_distortion.npy')
    tvecs4 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/tvecs4.npy')  
     
    mtxL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/left_camera.npy')
    distL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/left_distortion.npy')
    projL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/projection_matrixL.npy')
    rot_mat4 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/rotation_matrixL.npy')
    distL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/left_distortion.npy')
    tvecs3 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/tvecs3.npy')
    


    table_3d = table_3d[:, :2]
    for  i in range(4):
        
        point1 = table_3d[i]
        
        
        if i ==3 :
            point2 = table_3d[0]
        else:
            point2 = table_3d[i + 1]
        

        lines_table.append(sg.Segment2(sg.Point2(float(point1[0]), float(point1[1])), sg.Point2(float(point2[0]), float(point2[1]))))
    count_frames = 0
    
    while True:
        
        
        
        retR, imgR  = videoR.read()
        retL, imgL = videoL.read()
        
        imgR = cv2.undistort(imgR, mtxR, distR)
        imgL = cv2.undistort(imgL, mtxL, distL)

        trackR, imgR, featuresR = object_tracker.main(imgR, encoderR, trackerR, yoloR, class_names, camera_id = 0)
        trackL, imgL, featuresL = object_tracker.main(imgL, encoderL, trackerL, yoloL, class_names, camera_id = 1)
        
        
        

        

        
        if  not trackL.tracks or not trackR.tracks:
            continue
       
        for  i in range(4):
            
            point1 = x[i][0]
            
            
            if i ==3 :
                point2 = x[0][0]
            else:
                point2 = x[i + 1][0]
            
            
            cv2.line(imgR, (point1[0], point1[1]), (point2[0], point2[1]), (0, 255, 0), thickness = 3)
            
        

        detectionsL = [Detects(boxl.track_id, boxl.to_tlwh(), mtxL, rot_mat4, projL, tvecs4, distL, imgL) for boxl in trackL.tracks if boxl.is_confirmed() and boxl.time_since_update <= 1]
        detectionsR = [Detects(boxr.track_id, boxr.to_tlwh(), mtxR, rot_mat3, projR, tvecs3, distR, imgR) for boxr in trackR.tracks if boxr.is_confirmed() and boxr.time_since_update <= 1]
        
        # for i in detectionsL:
        #     cv2.rectangle(img, (i.x, i.y), (i.x + i.w, i.y + i.h), (0,255,0),2)
            
        #     cv2.rectangle(img_dinter,(i.x, i.y-45), (i.x + 100, i.y), (0, 255,0), -1)
        #     cv2.putText(img_dinter, f'ID:{i.id}', (i.x, i.y-5), 1, 3, (0, 0, 255), 2)
            
        # for i in detectionsR:
        #     cv2.rectangle(img_dinter, (i.x, i.y), (i.x + i.w, i.y + i.h), (0,255,0),2)
            
        #     cv2.rectangle(img_dinter,(i.x, i.y-45), (i.x + 100, i.y), (0, 255,0), -1)
        #     cv2.putText(img_dinter, f'ID:{i.id}', (i.x, i.y-5), 1, 3, (0, 0, 255), 2)
            
        frames = [imgR, imgL]
        boxes_3d = create_3d_bb.bb(frames, results, detectionsL, detectionsR, lines_table)
        boxes_3d = []
        if cv2.waitKey(1) & 0xFF==27:
            cv2.destroyAllWindows()
            break
        else:
            pass


    
    
    videoR.release()
    videoL.release()


