# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:00:48 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:58:20 2021

@author: Admin
"""

import cv2
import numpy as np


from XYZ_WFK import camera_pos, calcWFKfromPhotoMatrix, calcPhotogrammetryMatrixFromSolvePnp
import time
from collections import namedtuple

width = 5.37 ## mm 
height = 4.04 ## mm

Xpixel= width/1920 
Ypixel= height/1080 

pixel = (Xpixel + Ypixel) / 2

class Detects:
    
    diction = {}
    
    def __init__(self, 
                 num, 
                 box_list, 
                 
                 
                 camera_matrix,
                 rotation_matrix,
                 projection_matrix,
                 tvecs,
                 dist,
                 frame):
        
        self.x = box_list[0]
        self.y = box_list[1]
        self.w = box_list[2]
        self.h = box_list[3]
        self.list = box_list
        self.dist = dist
        self.frame = frame
        self.id = num
        
        # exterior openCV orientation
        self.rotation_matrix = rotation_matrix
        self.projection_matrix = projection_matrix
        self.tvecs = tvecs
        
        self.down_mid_x = (self.x + self.w / 2.0) 
        self.down_mid_y = (self.y + self.h) 
        
        self.camera_matrix = camera_matrix
        # interior orientation
        
        self.xo = (self.camera_matrix[0, 2] - 1920.0 / 2.0)
        self.yo = (-self.camera_matrix[1, 2] + 1080.0 / 2.0)
        
        self.f = ((self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2.0)
        
        
        # exterior photogrammetric orientation
        self.gXo, self.gYo, self.gZo = camera_pos(self.rotation_matrix, self.tvecs)
        
        # self.omega, self.phi, self.kappa = calcWFKfromPhotoMatrix(calcPhotogrammetryMatrixFromSolvePnp(self.rotation_matrix))
        
        self.photogram_rot_mat = calcPhotogrammetryMatrixFromSolvePnp(self.rotation_matrix)
        
        # ground coordinate system
        self.X = None
        self.Y = None
        
        self.down_mid_x_corrected = (self.down_mid_x - 1920.0 / 2.0)
        self.down_mid_y_corrected =  (1080.0 / 2.0 - self.down_mid_y )
        
        # print(self.down_mid_x_corrected, self.down_mid_y_corrected)
        
        # self.x_corrected = self.x - 1920 / 2
        # self.y_corrected = 1080/2 - self.y 
        
        
       
    def triangulation_z(self, z = 0.006):
        
        X = self.gXo + (z - self.gZo) * ((self.photogram_rot_mat[0, 0]*(self.down_mid_x_corrected - self.xo) + \
                                          self.photogram_rot_mat[1, 0]*(self.down_mid_y_corrected - self.yo) - \
                                          self.photogram_rot_mat[2, 0]*self.f) / (self.photogram_rot_mat[0, 2]*(self.down_mid_x_corrected - self.xo)+ \
                                                                                   self.photogram_rot_mat[1, 2]*(self.down_mid_y_corrected - self.yo)- \
                                                                                   self.photogram_rot_mat[2, 2]*self.f))
                                                                                   
        Y = self.gYo +    (z - self.gZo)* ((self.photogram_rot_mat[0, 1]*(self.down_mid_x_corrected - self.xo) + \
                                          self.photogram_rot_mat[1, 1]*(self.down_mid_y_corrected - self.yo) - \
                                          self.photogram_rot_mat[2, 1]*self.f) / (self.photogram_rot_mat[0, 2]*(self.down_mid_x_corrected - self.xo)+ \
                                                                                   self.photogram_rot_mat[1, 2]*(self.down_mid_y_corrected - self.yo)- \
                                                                                   self.photogram_rot_mat[2, 2]*self.f))                                                               
    
            
        self.X = X
        self.Y = Y 
        # print(self.X, self.Y)
        coordinates = np.float32(np.array([self.X, self.Y, z]))
        mid_down = np.int32(cv2.projectPoints(coordinates.reshape(1, -1), cv2.Rodrigues(self.rotation_matrix)[0], self.tvecs, self.camera_matrix, np.zeros((1,5)) )[0])     
        
    
        cv2.circle(self.frame, (mid_down[0][0][0], mid_down[0][0][1]), 10, (0, 0, 0), -1)
        # print(mid_down)
        return(self)

        

    def correlation_distance(self, detection):
        
        
        dist = np.sqrt((self.X - detection.X)**2 + (self.Y - detection.Y)**2)

        Detects.diction[self.id , detection.id] = dist
        

        
        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
            



     