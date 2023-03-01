import sys
import os
import argparse
import model.DenseDelay
from model.DenseDelay import DenseDelay

def main(args):
    if args.dbPath==None:
        sys.exit("""Please give the dataset path with the argument --dbPath "dataset/path" """)
    if os.path.exists(fr"{args.dbPath}/{args.type}/videoVectorSimilarity.npy")==False:
        sys.exit(f"""Videos not found, did you process the data set before testing it: "python process_db.py --dbPath {args.dbPath} --type{args.type}" """)

    denseDelay = DenseDelay.DenseDelay(args.type)
    denseDelay.loadWeights(fr"{args.weights}/{args.type}")
    denseDelay.evaluate(fr"{args.dbPath}/{args.type}/videoVectorSimilarity.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbPath", type=str, help="Path to the dataset that will be tested.")
    parser.add_argument("--weights", type=str, default="model\DenseDelay\weight", help="Weights use to evaluate the DenseDelay model.")
    parser.add_argument("--type", type=str, default="flows", help="Type of data evaluated(flows or images).")
    args = parser.parse_args()
    main(args)