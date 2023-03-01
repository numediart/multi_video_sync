import sys
import os
import argparse
import shutil
import random
import cv2 as cv
import numpy as np

class PreProcess:
    def __init__(self, dbPath: str):
        self.dbPath = fr"{dbPath}"

    def gray_resize(self, path: str):
        images = os.listdir(fr"{path}/good/left")
        dim = (224,224)
        for img in images:
            left = cv.imread(fr"{path}/good/left/{img}", cv.IMREAD_UNCHANGED)
            right = cv.imread(fr"{path}/good/right/{img}", cv.IMREAD_UNCHANGED)

            if len(left.shape)==3:
                left = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
            if len(right.shape)==3:
                right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

            left = cv.resize(left, dim, interpolation = cv.INTER_AREA)
            right = cv.resize(right, dim, interpolation = cv.INTER_AREA)

            cv.imwrite(fr"{path}/good/left/{img}", left)
            cv.imwrite(fr"{path}/good/right/{img}", right)
        
    def wrong_pair(self, path: str):
        if os.path.exists(fr"{path}/wrong/left")==False:
            os.makedirs(fr"{path}/wrong/left")
        if os.path.exists(fr"{path}/wrong/right")==False:
            os.makedirs(fr"{path}/wrong/right")

        images = os.listdir(fr"{path}/good/left")
        numberImages = len(images)
        if numberImages != len(os.listdir(fr"{path}/wrong/left")):
            for i in range(numberImages):
                frame = 0
                while(frame == 0 or f"{secondImage}.jpg" not in images):
                    frame = random.randint(-10,10)
                    secondImage = i + frame

                shutil.copyfile(fr"{path}/good/left/{i}.jpg", fr"{path}/wrong/left/{i}.jpg")
                shutil.copyfile(fr"{path}/good/right/{secondImage}.jpg", fr"{path}/wrong/right/{i}.jpg")

    def create_flow(self, path: str):
        if os.path.exists(fr"{path}/flows/good/left")==False:
            os.makedirs(fr"{path}/flows/good/left")
        if os.path.exists(fr"{path}/flows/good/right")==False:
            os.makedirs(fr"{path}/flows/good/right")

        numberImages = len(os.listdir(fr"{path}/images/good/left"))
        if len(os.listdir(fr"{path}/flows/good/left")) != numberImages-1:
            i = 0; error=0; listErrors = []

            frame = cv.imread(fr"{path}/images/good/left/{i}.jpg")
            prvsL = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

            frame = cv.imread(fr"{path}/images/good/right/{i}.jpg")
            prvsR = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame)
            hsv[...,1] = 255
            i+=1

            while i < numberImages:
                #Left
                frame = cv.imread(fr"{path}/images/good/left/{i}.jpg")
                nextL = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

                flow = cv.calcOpticalFlowFarneback(prvsL,nextL, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
                rgb_flowL = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

                rgb_flowL = cv.normalize(rgb_flowL, dst=None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

                #Right
                frame = cv.imread(fr"{path}/images/good/right/{i}.jpg")
                nextR = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

                flow = cv.calcOpticalFlowFarneback(prvsR,nextR, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
                rgb_flowR = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

                rgb_flowR = cv.normalize(rgb_flowR, dst=None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

                if(np.any(rgb_flowL != 0) and np.any(rgb_flowR != 0)):
                    cv.imwrite(fr"{path}/flows/good/left/{i-1}.jpg", rgb_flowL)
                    cv.imwrite(fr"{path}/flows/good/right/{i-1}.jpg", rgb_flowR)
                    prvsL = nextL
                    prvsR = nextR
                    i+=1
                else:
                    error+=1
                    if(error==20):
                        listErrors.append(i)
                        error=0
                        frame = cv.imread(fr"{path}/images/good/left/{i}.jpg")
                        prvsL = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                        frame = cv.imread(fr"{path}/images/good/right/{i}.jpg")
                        prvsR = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                        i+=1

            if len(os.listdir(fr"{path}/flows/good/left")) != numberImages-1:
                sys.exit(fr"""Errors appeared during the creation of the flows, it occurs frequently under Windows.
Please launch this command "python tool/process_image.py --dbPath "{self.dbPath}" --processImages 0" as long as you see this error then relaunch the original program.""")

    def start(self, process_images=True):
        print("Pre-process start !")
        if os.path.exists(fr"{self.dbPath}/images/good/left") and os.path.exists(fr"{self.dbPath}/images/good/right"):
            if process_images:
                print("Pre-process images...")
                self.gray_resize(fr"{self.dbPath}/images")
                self.wrong_pair(fr"{self.dbPath}/images")

            print("Pre-process flows...")
            self.create_flow(fr"{self.dbPath}")
            self.wrong_pair(fr"{self.dbPath}/flows")
            print("Pre-process ending ! ")
        else:
            sys.exit(fr"The path: {self.dbPath}/images/good/left or {self.dbPath}/images/good/right does not exist.")      

def main(args):
    if args.dbPath==None:
        sys.exit("""Please give the dataset path with the argument --dbPath "dataset/path". """)

    process = PreProcess(dbPath=args.dbPath)
    process.start(process_images=args.processImages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbPath", type=str, help="Path to the database that will be processed.")
    parser.add_argument("--processImages", type=int, default=True, help="Set to false to process only the flow (necessary in case of error).")
    args = parser.parse_args()

    main(args)