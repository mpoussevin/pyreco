import numpy as np
import logging
import sys
import argparse

import pyrecoutils


class OverallBias:
    def __init__(self):
        self.bias = 0

    def fit(self, reviews):
        self.bias = np.mean([r for u, i, r in reviews])
        logging.info(u"Overall bias is: %f", self.bias)


def main(input_filename, output_filename):
    reviews = pyrecoutils.load_ratings(input_filename)
    training_set, validation_set, test_set = pyrecoutils.split_sets(reviews)

    model = OverallBias()
    model.fit(training_set)

    with pyrecoutils.PredictionWriter(output_filename) as out:
        out.dump(training_set, [model.bias] * len(training_set))
        training_rmse = pyrecoutils.rmse(training_set, [model.bias] * len(training_set))
        out.space_line()
        out.dump(validation_set, [model.bias] * len(validation_set))
        validation_rmse = pyrecoutils.rmse(validation_set, [model.bias] * len(validation_set))
        out.space_line()
        out.dump(test_set, [model.bias] * len(test_set))
        test_rmse = pyrecoutils.rmse(test_set, [model.bias] * len(test_set))
    logging.info(u"RMSE: %.4f %.4f %.4f", training_rmse, validation_rmse, test_rmse)


if __name__ == u"__main__":
    logging.basicConfig(format=u'%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
    logging.info(u"Called: %s", u" ".join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(u"input", help=u"Path to the input file.", type=unicode)
    parser.add_argument(u"output", help=u"Path to the output file.", type=unicode)
    args = parser.parse_args()
    main(args.input, args.output)
	