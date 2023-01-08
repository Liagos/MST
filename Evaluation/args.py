import argparse
from configparser import ConfigParser


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--configParserFile", dest='config_file', default='config.ini', type=str,
                        help="path to config file for the annotation parser")

    args = parser.parse_args()

    config_file = args.config_file
    config = ConfigParser()
    config.read(config_file)

    parser.add_argument("--runsPath", dest="runsPath", default=config["runsPath"]["path"], type=str,
                        help="path to latest runs file")
    parser.add_argument("--yoloPath", dest="yoloPath", default=config["yoloPath"]["path"], type=str,
                        help="path to yolo repository")
    parser.add_argument("--yamlFiles", dest="yamlFiles", default=config["yamlFiles"]["yaml"], type=str,
                        help="yaml files needed for validation")
    parser.add_argument("--yamlTrain", dest="yamlTrain", default=config["yamlTrain"]["yaml_train"], type=str,
                        help="yaml file needed for training")
    parser.add_argument("--testData", dest="testData", default=config["testData"]["path"], type=str,
                        help="paths to COCO, FES, and ICARUS test data")
    parser.add_argument("--evalPath", dest="evalPath", default=config["evalPath"]["path"], type=str,
                        help="path to evaluation script results")
    parser.add_argument("--dictsPath", dest="dictsPath", default=config["dictsPath"]["path"], type=str,
                        help="path to save dicts")
    parser.add_argument("--numEpochs", dest="numEpochs", default=config["numEpochs"]["num"], type=int,
                        help="number of epochs")
    parser.add_argument("--batchSize", dest="batchSize", default=config["batchSize"]["num"], type=int,
                        help="number of batch size")

    return parser.parse_known_args()[0] if known else parser.parse_args()
