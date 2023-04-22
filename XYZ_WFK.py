import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation as Rotate

""" Calculates rotation matrix to euler angles
The result is the same as MATLAB except the order
of the euler angles (x and z are swapped) """
    
def camera_pos(R, tvec):

    R = R.transpose()
    cameraPosition= -np.matrix(R) * np.matrix(tvec)
    
    return(cameraPosition)

def opencvRotationMatrixToEulerAngles(R) :

    # assert(isRotationMatrix(R))
    x_angle = np.arctan2(R[1 ,0], R[0, 0])
    y_angle = np.arcsin(-R[2, 0])
    z_angle = np.arctan2(R[2, 1], R[2,2])
    
    return(180*x_angle/np.pi, 180*y_angle/np.pi, 180*z_angle/np.pi)

def calcPhotogrammetryMatrixFromSolvePnp(R):

    Rot = np.copy(R)
    Rot[1, 0] = -Rot[1, 0]
    Rot[1, 1] = -Rot[1, 1]
    Rot[1, 2] = -Rot[1, 2]
    Rot[2, 0] = -Rot[2, 0]
    Rot[2, 2] = -Rot[2, 2]
    Rot[2, 1] = -Rot[2, 1]
    
    return(Rot)

# def rvecsFromPhotogrammetryMatrix(R):
#     w, f, k =  - cv2.Rodrigues(R.T)[0]
#     return(np.rad2deg(w), np.rad2deg(f), np.rad2deg(k))

def calcWFKfromPhotoMatrix(R):
    
    phi = np.arcsin(R[2, 0])
    sinOmega = -R[2, 1] / np.cos(phi)
    cosOmega = R[2, 2] / np.cos(phi)
    if cosOmega != 0:
        omegaAbs = np.degrees(np.abs(np.arctan(sinOmega/cosOmega)))

    if (sinOmega > 0.0 and cosOmega > 0.0):
        omega = omegaAbs

    if (sinOmega > 0.0 and cosOmega < 0.0):
        omega = 180.0 - omegaAbs

    if (sinOmega < 0.0 and cosOmega > 0.0):
        omega = -omegaAbs

    if (sinOmega < 0.0 and cosOmega < 0.0):
        omega = -(180.0 - omegaAbs)
        
    if (sinOmega == 1):
        omega = 90.0

    if (cosOmega == 1):
        omega = 0.0

    if (sinOmega == -1):
        omega = -90.0

    if (cosOmega == -1):
        omega = 180.0
        
    cosKappa = R[0, 0]/np.cos(phi);  
    sinKappa = -R[1, 0] / np.cos(phi)
    
    if (cosKappa != 0.0):
        kappaAbs = np.abs(np.rad2deg(np.arctan(sinKappa/cosKappa)))

    if (sinKappa > 0.0 and cosKappa > 0.0):
        kappa = kappaAbs
    
    if (sinKappa > 0.0 and cosKappa < 0.0):
        kappa = 180.0 - kappaAbs
    
    if (sinKappa < 0.0 and cosKappa > 0.0):
      kappa = -kappaAbs
    
    if (sinKappa < 0.0 and cosKappa < 0.0):
        kappa = -(180.0 - kappaAbs)
    
    if (sinKappa == 1):
        kappa = 90.0
    
    if (cosKappa == 1):
        kappa = 0.0
    
    if (sinKappa == -1):
        kappa = -90.0
    
    if (cosKappa == -1):
        kappa = 180.0

    phi = np.degrees(phi)
    if phi < 0 :
        phi = 360 - np.abs(phi)

    if omega < 0:
        omega = 360 - np.abs(omega)
  
    if kappa < 0 :
        kappa = 360 - np.abs(kappa)

    return(omega, phi, kappa)

if __name__ == '__main__':

    # rotationL = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/rotation_matrixL.npy')
    rotationR = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/rotation_matrixR.npy')
    
    # tvecs4 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/left_camera_info/tvecs4.npy')
    tvecs3 = np.load('C:/Users/vero pc/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ_lastTry/right_camera_info/tvecs3.npy')
        
    # cam_Position4 = camera_pos(rotationL, tvecs4)
    cam_Position3 = camera_pos(rotationR, tvecs3)
    
    # np.save("cam_Position4.npy", cam_Position4)
    np.save("cam_Position3.npy", cam_Position3)

    photo_matrix = calcPhotogrammetryMatrixFromSolvePnp(rotationR)
    wmega, phi, kappa = calcWFKfromPhotoMatrix(photo_matrix)