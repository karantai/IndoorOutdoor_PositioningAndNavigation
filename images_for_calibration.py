import numpy as np
import cv2
import time





def printscreen(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        
        cv2.imwrite(f'calib_image_{counter}.tiff', frame)
        

if __name__=="__main__":
    cap=cv2.VideoCapture(0)
    # print(cap.get(3),cap.get(4))
    # WIDTH=cap.get(3)
    # HEIGHT=cap.get(4)
    # cap.set(3,1080)
    # cap.set(4,720)
    
    counter = 0


    while (True):
         
        
        
        ret, frame = cap.read()
        
        
        
        
        if ret == True:

            counter +=1 
            cv2.namedWindow("test",cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('test', printscreen)
            cv2.imshow("test",frame)

            

            if cv2.waitKey(1) & 0xFF==ord('q'):
                
                print("Exiting program . . .")
                break
            
        else:
            print("End of VideoStream")
            break

    cap.release()

    cv2.destroyAllWindows()
