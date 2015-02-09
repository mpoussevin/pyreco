import numpy as np
import logging
import sys
import argparse

import pyrecoutils


class UserBias:
    def __init__(self, threshold=0):
        self.overall = 0
        self.biases = {}
        self.counts = {}
        self.threshold = threshold

    def fit(self, reviews):
        self.overall = np.mean([r for u, i, r in reviews])
        for u, i, r in reviews:
            self.biases[u] = self.biases.get(u, 0) + r
            self.counts[u] = self.counts.get(u, 0) + 1

    def select(self, reviews):
        logging.info(u"Selecting threshold on %d reviews", len(reviews))
        rmses = []
        for t in range(10):
            self.threshold = t
            predictions = [self.predict(u) for u, i, r in reviews]
            rmses.append(pyrecoutils.rmse(reviews, predictions))
        threshold = np.argmin(rmses)
        logging.info(u"Selected threshold: %f", threshold)
        self.threshold = threshold

    def predict(self, user):
        if self.counts.get(user, 0) > self.threshold:
            return self.biases[user] / self.counts[user]
        else:
            return self.overall


def main(input_filename, output_filename):
    reviews = pyrecoutils.load_ratings(input_filename)
    training_set, validation_set, test_set = pyrecoutils.split_sets(reviews)

    model = UserBias()
    model.fit(training_set)
    model.select(validation_set)

    with pyrecoutils.PredictionWriter(output_filename) as out:
        training_prediction = [model.predict(u) for u, i, r in training_set]
        out.dump(training_set, training_prediction)
        training_rmse = pyrecoutils.rmse(training_set, training_prediction)
        out.space_line()

        validation_prediction = [model.predict(u) for u, i, r in validation_set]
        out.dump(validation_set, validation_prediction)
        validation_rmse = pyrecoutils.rmse(validation_set, validation_prediction)
        out.space_line()

        test_prediction = [model.predict(u) for u, i, r in test_set]
        out.dump(test_set, test_prediction)
        test_rmse = pyrecoutils.rmse(test_set, test_prediction)
    logging.info(u"RMSE: %.4f %.4f %.4f", training_rmse, validation_rmse, test_rmse)


if __name__ == u"__main__":
    logging.basicConfig(format=u'%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
    logging.info(u"Called: %s", u" ".join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(u"input", help=u"Path to the input file.", type=unicode)
    parser.add_argument(u"output", help=u"Path to the output file.", type=unicode)
    args = parser.parse_args()
    main(args.input, args.output)
	