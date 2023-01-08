import argparse
from configparser import ConfigParser


def parse_opt_inference(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--configParserFile", dest='config_file', default='config_infer.ini', type=str,
                        help="path to config file for inference arguments")

    args = parser.parse_args()

    config_file = args.config_file
    config = ConfigParser()
    config.read(config_file)

    parser.add_argument("-m", "--modelname", dest="modelname", default=config["modelname"]["path"], type=str,
                        help="path to model weights")
    parser.add_argument("-t", "--testimages", dest="testimages", default=config["testimages"]["path"], type=str,
                        help="path to test images")
    parser.add_argument("-l", "--testlabels", dest="testlabels", default=config["testlabels"]["path"], type=str,
                        help="path to test labels")
    parser.add_argument("-o", "--outputDir", dest="outputDir", default=config["outputDir"]["path"], type=str,
                        help="path to output directory")

    return parser.parse_known_args()[0] if known else parser.parse_args()
