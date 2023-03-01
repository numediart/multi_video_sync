import sys
import argparse
from tool.process_image import PreProcess
from tool.process_video import *

def main(args):
    if args.dbPath==None:
        sys.exit("""Please give the dataset path with the argument --dbPath "dataset/path" """)

    if args.processImage==True:
        process = PreProcess(dbPath=args.dbPath)
        process.start()

    if args.processVideo==True:
        createVideos(dbPath=args.dbPath, type=args.type, pathReferenceList=args.pathReferenceList, numberVideos=args.numberVideo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbPath", type=str, help="Path to the dataset that will be used to train models.")
    parser.add_argument("--type", type=str, default="flows")
    parser.add_argument("--processImage", type=int, default=True, help="Set to true or false if it is necessary to pre-process images before evaluating them.")
    parser.add_argument("--processVideo", type=int, default=True, help="Set to true or false if it is necessary to create videos that will be evaluated.")
    parser.add_argument("--pathReferenceList", type=str, default="", help="Path to video references to create videos according to a certain schema (useful to create videos with similar image and flow to compare them)")
    parser.add_argument("--numberVideo", type=int, default=3000, help="Number of video that will be created.")
    args = parser.parse_args()
    main(args)