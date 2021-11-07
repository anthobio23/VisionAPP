#!/usr/bin/python3.9

import numpy as np
from cv2 import cv2 as cv

class Recognition_img:
    
    def __init__(self, img):

        self.img = img
        pass

    def Processing_CV2(self):

        # Convertion to scales Grays
        Grays = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        # Convertion to threshold
        _, threshold = cv.threshold(Grays, 100, 255, cv.THRESH_BINARY)

        #give to image contour
        contour, hierarchy = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        # pain countours
        IMG_CONTOUR = cv.drawContours(self.img, contour, -1, (0, 255, 0), 3)

        cv.imshow("Contour in image", IMG_CONTOUR)
        pass

    def Processing_CV2_other(self):

        #[0, 255] -> [0, 1]
        Grays = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        
        # gauss
        Gauss = cv.GaussianBlur(src=Grays, ksize=(3, 3), sigmaX=3)

        Canny = cv.Canny(Gauss, 60, 100)
        cv.imshow("", Canny)
        
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)

        close = cv.morphologyEx(src=Canny, op=cv.MORPH_CLOSE, kernel=kernel)

        contours, hierarchy = cv.findContours(close.copy(), 
                cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # print to amount
        print(f"print to money: {len(contours)}")

        cv.drawContours(self.img, 
                contours,
                -1, 
                (0, 0, 255))
        pass

    # show camera
    def camera_show(self):

        CaptureVideo = cv.VideoCapture(0)
        if not CaptureVideo.isOpened(): # if the window of camera not open

            print("not Camera")
            exit() # exit of run
            pass
        while True: # while detect camera

            TypeCamera, Camera = CaptureVideo.read() # read to camera
            Grays = cv.cvtColor(Camera, cv.COLOR_BGR2GRAY) # transform to scale grays
            cv.imshow("in live", Grays)
            
            # Exit with "q"
            if  cv.waitKey(1) == ord("q"): 
                break
            pass

        CaptureVideo.release()
        pass
    pass


# insert a image
PATH = input("PATH of image: ")
image = cv.imread(filename=PATH)

if __name__ == "__main__":
    
    VISION_COMP = Recognition_img(img=image)

    if PATH == "contorno.jpg":

        VISION_COMP.Processing_CV2()
        cv.waitKey(0) # Image static
        cv.destroyAllWindows()

    elif PATH == "v":
        VISION_COMP.camera_show()
        cv.destroyAllWindows()

    else:

        VISION_COMP.Processing_CV2()
        cv.waitKey(0) # Image static
        cv.destroyAllWindows()
    pass

