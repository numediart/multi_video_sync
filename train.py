import sys
import os
import argparse
from model.TripletLossEuc import TripletLossEuc
from model.DenseDelay import DenseDelay
from tool.process_image import PreProcess
from tool.process_video import *

def main(args):
    if args.dbPath==None:
        sys.exit("""Please give the dataset path with the argument --dbPath "dataset/path" """)

    process = PreProcess(dbPath=args.dbPath)
    process.start()

    similarityModel = TripletLossEuc.TripletLossEuc(args.type)
    similarityModel.train(args.dbPath)

    createVideos(args.dbPath, args.type, args.pathReferenceList, args.numberVideo)

    denseDelay = DenseDelay.DenseDelay(args.type)
    denseDelay.train(fr"{args.dbPath}/{args.type}/videoVectorSimilarity.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbPath", type=str, help="Path to the dataset that will be used to train models.")
    parser.add_argument("--type", type=str, default="flows", help="Type of data for the training(flows or images).")
    parser.add_argument("--pathReferenceList", type=str, default="", help="Numpy array of video references if already created.")
    parser.add_argument("--numberVideo", type=int, default=40000, help="Number of videos created for train DenseDelay.")
    args = parser.parse_args()
    main(args)