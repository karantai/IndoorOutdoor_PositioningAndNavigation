
import numpy as np
import cv2
# import decompose

import pandas as pd
# import undistort_RATIONAL_without_otpimal



def coords( event, x, y, flags, param ):
    global i
    global j
    global k 
    global l
    
    
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(frame, f'{param[k]}', (x-10, y-10),  cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
        i += 1
        k+=1
        print(k)

        if i <= len(param_[0]):
            imgp4[i] = (x,y)
            print(x,y)

            
        else:
            j+=1
            imgp3[j] = (x,y)



def pose(Coords, imgp, mtx, dst = np.zeros((1,5))):
    
    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv2.solvePnP(Coords, imgp, mtx, dst)
    
    return (rvecs, tvecs)



def make_projMatrix(rvecs, tvecs, camera_mtx):
    
    rotation_matrix = cv2.Rodrigues(rvecs)[0]
    projection_matrix = np.append(rotation_matrix, tvecs, axis=1)
    projection_matrixCam = np.dot(camera_mtx, projection_matrix)
    
    
    return(rotation_matrix, projection_matrixCam)


def ext_points_test(cam_points):
    points_coords=[]
    
    points_list = cam_points.strip().split(',') 
 
    for i in points_list:
        points_coords.append(tuple(Coords.iloc[int(i)-1,1:]))
        fixed_points4pose = np.array(points_coords, np.float64).reshape(-1, 1, 3)
        dict_points_info = dict(zip(points_list, points_coords))
        print('Good Job')
     
    return(dict_points_info, fixed_points4pose)
    
def exteriorOrient_points(imgp):

    imgp = [i for i in imgp.values()]
    imgp = np.array(imgp, np.float64())
    imgp4pose = imgp.reshape(-1, 1, 2)

    return(imgp4pose)

def exteriorOrient_points2(imgp):
    
    
    imgp = [i for i in imgp.values()]
    imgp = np.array(imgp, np.float64())
    imgp = imgp.T

    
    return(imgp)

def undistort(frame, mtx, dist):
    frame = cv2.undistort(frame, mtx, dist)
    return(frame)


if __name__=='__main__':

    video_left = cv2.VideoCapture('left_video.mp4')
    video_right = cv2.VideoCapture('right_video.mp4')

    Coords = pd.read_csv("coords.txt", sep = ",", header = None)
    
    # Start Live video

    i = 0 
    j = 0
    k = 0

    
    # sensor's width and height in mm
    width = 5.37
    height = 4.04
    Xpixel = width/1920
    Ypixel = height/1080
    pixel = (Xpixel+Ypixel)/2

    global imgp3
    imgp3 = {}
    global imgp4
    imgp4 = {}

    camera3_mtx = np.load('right_camera_info/right_camera.npy')
    camera3_distortion = np.load('right_camera_info/right_distortion.npy')
    
    camera4_mtx = np.load('left_camera_info/left_camera.npy')
    camera4_distortion = np.load('left_camera_info/left_distortion.npy')
    
    cam4_points = input('Give points for exterior orientation for the left camera seperated with comma: ')
    fixed_points4, fixed_points4pose4 = ext_points_test(cam4_points)
    cam3_points = input('Give points for exterior orientation for the right camera seperated with comma: ')    
    fixed_points3, fixed_points4pose3 = ext_points_test(cam3_points)
    
    param_ = (list(map(int,cam4_points.split(','))), list(map(int, cam3_points.split(','))))
    flat_param=[]
    for m in param_:
        for n in m:
            flat_param.append(n)
            

    while(1):
         # Capturing frames
        ret4, frame4 = (lambda video_left : video_left.read())(video_left)
        ret3, frame3 = (lambda video_right : video_right.read())(video_right)



        frames = [cv2.undistort(np.float64(frame4), camera4_mtx, camera4_distortion),\
                  cv2.undistort(np.float64(frame3), camera3_mtx, camera3_distortion)]
        
        # frames = [frame4, frame3]

        for frame in frames:
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', coords, flat_param)
            
            
      
            while (True):
                cv2.imshow('image',frame/255.0)

                if cv2.waitKey(1) & 0xFF==27:
                    cv2.destroyAllWindows()
                    break
                else:
                    pass

        break

    imgp4poseL = exteriorOrient_points(imgp4)
    np.save("image_points4poseL", imgp4poseL)
   
    imgp4poseR = exteriorOrient_points(imgp3)
    np.save("image_points4poseR",imgp4poseR)
    
    rvecs4, tvecs4 = pose(fixed_points4pose4, imgp4poseL, camera4_mtx)
    rvecs3, tvecs3 = pose(fixed_points4pose3, imgp4poseR, camera3_mtx)
    
    rotation_matrixR, projection_matrixR = make_projMatrix(rvecs3, tvecs3, camera3_mtx)
    np.save("projection_matrixR.npy", projection_matrixR)
    np.save("rotation_matrixR.npy", rotation_matrixR)
    np.save("tvecs3.npy", tvecs3)
    
    
    rotation_matrixL, projection_matrixL = make_projMatrix(rvecs4, tvecs4, camera4_mtx)
    np.save("projection_matrixL.npy", projection_matrixL)
    np.save("rotation_matrixL.npy", rotation_matrixL)
    np.save("tvecs4.npy", tvecs4)
    
 

    cv2.destroyAllWindows()
        