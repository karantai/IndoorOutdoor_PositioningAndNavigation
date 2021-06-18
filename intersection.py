#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:47:42 2020

@author: karas
"""

import numpy as np
import cv2
from XYZ_WFK import camera_pos, calcWFKfromPhotoMatrix, calcPhotogrammetryMatrixFromSolvePnp
from solve_equations import solveObservationEquations
import time
from collections import namedtuple
# from numba import jit


width = 5.37 ## mm 
height = 4.04 ## mm
Xpixel= width/1920 
Ypixel= height/1080 
pixel = (Xpixel + Ypixel) / 2








class Intersect:
    
    
    _registry = []
    
    ### objects == center of boxes with attributes with respect to their cameras
    
    
    
    def __init__(self, 
                 camera_matrix,
                 rotation_matrix,
                 tvecs,
                 projection_matrix,
                 list_points):
        
        
        
        ## list_points in form of [ L/R, PID, X, Y ] 
        
        self.camera = str(list_points[0])
        self.PID = list_points[1]
        
        # center of the bounding box
        self.cx = list_points[2]
        self.cy = list_points[3]
        
        self.camera_matrix = camera_matrix
        self.rotation_matrix = rotation_matrix
        self.projection_matrix = projection_matrix
        self.tvecs = tvecs
        
        self._registry.append(self)
        
        self.list_points = list_points
        
        


        
        self.gXo, self.gYo, self.gZo = camera_pos(self.rotation_matrix, self.tvecs)
        self.omega, self.phi, self.kappa = calcWFKfromPhotoMatrix(calcPhotogrammetryMatrixFromSolvePnp(self.rotation_matrix))
        
    
    ## this method returns the string representation of the object. 
    ##This method is called when print() or str() function is invoked on an object.
    ##This method must return the String object. 
    ##If we don’t implement __str__() function for a class, then built-in object 
    ##implementation is used that actually calls __repr__() function.
    def __str__(self):
        
        return(f'camera : {self.camera}\nID : {self.PID}\nx : {self.cx}\ny : {self.cy}')
    
    ##Python __repr__() function returns the object representation. 
    ##It could be any valid python expression such as tuple, dictionary, string etc.

    ##This method is called when repr() function is invoked on the object, 
    ##in that case, __repr__() function must return a String otherwise error will be thrown.
    
    def __repr__(self):
        return(f'[{self.camera}, {self.PID}, {self.cx}, {self.cy}]')
    
    
    
    
    
        
        
     
      
    ## n is the the number of points in x, y    
    
    def designEquationMatrices(p1, p2):
        
        

        n = 2
        
        
        ## matrices for least square algorithm
        mA = np.zeros((2*n, 3), dtype = np.float64)
        mB = np.zeros((2*n, 1), dtype = np.float64)
        mXo = np.zeros((3, 1), dtype = np.float64)
        coords = cv2.triangulatePoints(p1.projection_matrix, p2.projection_matrix,
                                            np.array(p1.list_points[2:], dtype = np.float64),
                                                     np.array(p2.list_points[2:], dtype = np.float64))
        
        gX, gY, gZ = coords[:3]/coords[3]


        
        p = [p1, p2]
        counter = 0
        loop = 0
        
        for  i in p:
            
            
            
            # pixels
            # px = 1920/2 + i.camera_matrix[0,2] + i.cx
            # py = 1080/2 - i.camera_matrix[1,2] - i.cy
            
            
            px = i.cx - 1920/2.0
            py =  - i.cy + 1080/2.0
            
            xo = i.camera_matrix[0,2] - 1920/2.0
            yo = - i.camera_matrix[1,2] + 1080/2.0
            
            
            
            
            
            f = (i.camera_matrix[0,0] + i.camera_matrix[1,1])/2
            
            

            # meters
            
    
            mXo[0, 0] = gX
            mXo[1, 0] = gY
            mXo[2, 0] = gZ
        
            
            ### R katharoi arithmoi ( gwnies se rad )
            R11 = np.cos(i.kappa) * np.cos(i.phi)
            R12 = np.cos(i.kappa) * np.sin(i.phi) * np.sin(i.omega) + np.sin(i.kappa) * np.cos(i.omega)
            R13 = -np.cos(i.kappa) * np.sin(i.phi) * np.cos(i.omega) + np.sin(i.kappa) * np.sin(i.omega)
            R21 = -np.sin(i.kappa) * np.cos(i.phi)
            R22 = -np.sin(i.kappa) * np.sin(i.phi) * np.sin(i.omega) + np.cos(i.kappa) * np.cos(i.omega)
            R23 = np.sin(i.kappa) * np.sin(i.phi) * np.cos(i.omega) + np.cos(i.kappa) * np.sin(i.omega)
            R31 = np.sin(i.phi)
            R32 = -np.cos(i.phi) * np.sin(i.omega)
            R33 = np.cos(i.phi) * np.cos(i.omega)
            
            # U, V, W SE METRA
            U = R11 * (gX - i.gXo) + R12 * (gY - i.gYo) + R13 * (gZ - i.gZo)
            V = R21 * (gX - i.gXo) + R22 * (gY - i.gYo) + R23 * (gZ - i.gZo)
            W = R31 * (gX - i.gXo) + R32 * (gY - i.gYo) + R33 * (gZ - i.gZo)
            
            # meters2
            W2 = np.power(W, 2)
            
            
            mA[counter, 0] = f * ((R31 * U - R11 * W) / W2)
            mA[counter, 1] = f * ((R32 * U - R12 * W) / W2)
            mA[counter, 2] = f * ((R33 * U - R13 * W) / W2)
        
            
            mB[counter, 0] = px - (xo - f * U / W)
            
            counter += 1 
            
            mA[counter, 0] = f * ((R31 * V - R21 * W) / W2)
            mA[counter, 1] = f * ((R32 * V - R22 * W) / W2)
            mA[counter, 2] = f * ((R33 * V - R23 * W) / W2)
        
            mB[counter, 0] =  py - (yo - f * V / W)
            
            counter += 1



        mP = np.identity(2*n)
        
        free = mA.shape[0] - 3



        
        
        info = solveObservationEquations(mA, mB, mP, mXo, free, gX, gY, gZ)
        
        loop += 1
        

        

        
        
        
        
        
        wrapper = namedtuple('matrices_info', ['mA', 'mB', 'mN', 'mu', 'mx', 
                            'mxa', 'mV', 'mCx', 'mCv', 'mCya', 'sigma', 'f',
                            'mXo', 'accuracy', 'x_accuracy', 'y_accuracy',
                            'z_accuracy'])
        
        
        
        matrices_info = wrapper(info[0], info[1], info[2], info[3], info[4],
                                info[5], info[6], info[7], info[8], info[9],
                                info[10], info[11], info[12],
                                info[13], info[14], info[15], info[16])
        
        
        
        

        
        
        
        

        
        
        
        

        
        
        
        return(matrices_info)
        
        
        

    
    

        
        
    
    
    
    
    
    
    
