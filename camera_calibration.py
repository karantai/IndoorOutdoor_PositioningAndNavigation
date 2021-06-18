import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
error_list = []

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)*10                        
              
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('left_camera_info/left_calib/*.jpg')
counter = 0 
for fname in images:
    
    img = cv2.imread(fname)
    print(img.shape)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,9), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        counter += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 9), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)                    
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

for i in images:
    und = cv2.undistort(cv2.imread(i), mtx, dist)
    cv2.imshow('img',und)
    cv2.waitKey(500)
print(f'Image counter : {counter}')
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    error_list.append(error)
    mean_error += error


print (f'Reprojection error : {mean_error/len(objpoints)}')
np.save('left_distortion.npy', dist)
mtx = 3 * mtx
mtx[2,2] = 1
np.save('left_camera.npy', mtx)
photos_list = [i+1 for i in range(counter)]
plt.bar(photos_list, error_list)
plt.title('Pattern view error')
plt.xlabel('Photo number')
plt.ylabel('error (pixels')
plt.axhline(y = mean_error/len(objpoints), color = 'red')
plt.show()              