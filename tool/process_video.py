import sys
import os
import argparse
import numpy as np
import random

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from model.TripletLossEuc import TripletLossEuc

def createVideos(dbPath, type, pathReferenceList="", numberVideos=3000):
    if pathReferenceList == "":
        numberImages = len(os.listdir(fr"{dbPath}/flows/good/left"))
        lenVideos = 20 # Length of a video in frame
        rng = 19 # Max delay of a video in frame

        data = []
        for _ in range(numberVideos):
            firstFrameLeft = random.randint(rng, numberImages-(lenVideos*2))
            delay = random.randint(-rng, rng+1)
            firstFrameRight = firstFrameLeft+delay

            data.append([firstFrameLeft, firstFrameRight, delay])

        np.save(fr"{dbPath}/videoReference", data)

    data = np.load(fr"{dbPath}/videoReference.npy")

    similarityModel = TripletLossEuc.TripletLossEuc(type=type)
    similarityModel.loadWeights(fr"model/TripletLossEuc/weight/{type}/weights")
    leftFeatures, rightFeatures = similarityModel.extractFeatures(dbPath)

    similarityData = []
    counter = 0

    for item in data:
        if counter%1000==0: print(counter)
        counter+=1

        featureSiameseLeft = np.array(leftFeatures[item[0]:item[0]+20])
        featureSiameseRight = np.array(rightFeatures[item[1]:item[1]+20])

        results = []
        for i in featureSiameseLeft:
            for j in featureSiameseRight:
                results.append(similarityModel.computeSimilarity(i, j))
                
        similarityData.append([results, item[0], item[1], item[2]])

    similarityData = np.array(similarityData, dtype=object)
    np.save(fr"{dbPath}/{type}/videoVectorSimilarity", similarityData)

def main(args):
    createVideos(args.dbPath, args.type, args.pathReferences, args.numberVideos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbPath", type=str, help="Path to the dataset that will be processed.")
    parser.add_argument("--type", type=str, default="flows", help="Type of video we want to create (flows or images).")
    parser.add_argument("--pathReferences", type=str, default="", help="Numpy array of video references if already created.")
    parser.add_argument("--numberVideos", type=int, default=3000, help="Number of videos that will be created.")
    args = parser.parse_args()

    if(args.dbPath==None): sys.exit("Pass an argument for the database. -h to get help.")
    main(args)