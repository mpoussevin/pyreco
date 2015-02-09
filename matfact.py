import numpy as np
import logging
import sys
import argparse

import pyrecoutils


class MatrixFactorization:
    def __init__(self, components, epochs, learning_rate, decay_rate, user_l2, item_l2, seed):
        # Overall
        self.overall = 0
        self.k = components
        np.random.seed(seed)
        # User parameters
        self.user_biases = {}
        self.user_counts = {}
        self.user_latent = {}
        # Item parameters
        self.item_biases = {}
        self.item_counts = {}
        self.item_latent = {}
        # Learning parameters
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.user_l2 = user_l2
        self.item_l2 = item_l2
        self.user_threshold = 0
        self.item_threshold = 0

    def _initialize(self, reviews):
        self.overall = np.mean([r for u, i, r in reviews])
        for u, i, r in reviews:
            self.user_biases[u] = np.random.randn()
            self.user_counts[u] = self.user_counts.get(u, 0) + 1
            self.user_latent[u] = np.random.randn(self.k)
            self.item_biases[i] = np.random.randn()
            self.item_counts[i] = self.item_counts.get(i, 0) + 1
            self.item_latent[i] = np.random.randn(self.k)

    def fit(self, training_reviews, validation_reviews, test_reviews):
        self._initialize(training_reviews)
        for epoch in range(self.epochs):
            for u, i, r in training_reviews:
                b_u, b_i = self.user_biases[u], self.item_biases[i]
                p_u, q_i = self.user_latent[u], self.item_latent[i]
                delta = self.overall + b_u + b_i + np.dot(p_u, q_i) - r
                self.user_biases[u] -= self.learning_rate * (delta + self.user_l2 * b_u)
                self.user_latent[u] -= self.learning_rate * (delta * q_i + self.user_l2 * p_u)
                self.item_biases[i] -= self.learning_rate * (delta + self.item_l2 * b_i)
                self.item_latent[i] -= self.learning_rate * (delta * p_u + self.item_l2 * q_i)
            self.learning_rate *= self.decay_rate
            training_rmse = pyrecoutils.rmse(training_reviews, [self.predict(u, i) for u, i, r in training_reviews])
            validation_rmse = pyrecoutils.rmse(validation_reviews,
                                               [self.predict(u, i) for u, i, r in validation_reviews])
            test_rmse = pyrecoutils.rmse(test_reviews, [self.predict(u, i) for u, i, r in test_reviews])
            logging.info(u"Epoch % 4d - RMSE %f %f %f", epoch, training_rmse, validation_rmse, test_rmse)

    def select(self, reviews):
        logging.info(u"Selecting threshold on %d reviews", len(reviews))
        rmses = []
        for t_u in range(10):
            self.user_threshold = t_u
            for t_i in range(10):
                self.item_threshold = t_i
                predictions = [self.predict(u, i) for u, i, r in reviews]
                rmses.append(pyrecoutils.rmse(reviews, predictions))
        threshold = np.argmin(rmses)
        user_threshold = threshold / 10
        item_threshold = threshold % 10
        logging.info(u"Selected user threshold: %f", user_threshold)
        self.user_threshold = user_threshold
        logging.info(u"Selected item threshold: %f", item_threshold)
        self.item_threshold = item_threshold

    def predict(self, user, item):
        prediction = self.overall
        if self.user_counts.get(user, 0) > self.user_threshold and self.item_counts.get(item, 0) > self.item_threshold:
            prediction += self.user_biases[user] + self.item_biases[item]
            prediction += np.dot(self.user_latent[user], self.item_latent[item])
        return prediction


def main(input_filename, output_filename, components, epochs, learning_rate, decay_rate, user_l2, item_l2, seed):
    reviews = pyrecoutils.load_ratings(input_filename)
    training_set, validation_set, test_set = pyrecoutils.split_sets(reviews)

    model = MatrixFactorization(components, epochs, learning_rate, decay_rate, user_l2, item_l2, seed)
    model.fit(training_set, validation_set, test_set)
    model.select(validation_set)

    with pyrecoutils.PredictionWriter(output_filename) as out:
        training_prediction = [model.predict(u, i) for u, i, r in training_set]
        out.dump(training_set, training_prediction)
        training_rmse = pyrecoutils.rmse(training_set, training_prediction)
        out.space_line()

        validation_prediction = [model.predict(u, i) for u, i, r in validation_set]
        out.dump(validation_set, validation_prediction)
        validation_rmse = pyrecoutils.rmse(validation_set, validation_prediction)
        out.space_line()

        test_prediction = [model.predict(u, i) for u, i, r in test_set]
        out.dump(test_set, test_prediction)
        test_rmse = pyrecoutils.rmse(test_set, test_prediction)
    logging.info(u"RMSE: %.4f %.4f %.4f", training_rmse, validation_rmse, test_rmse)


if __name__ == u"__main__":
    logging.basicConfig(format=u'%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
    logging.info(u"Called: %s", u" ".join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(u"input", help=u"Path to the input file.", type=unicode)
    parser.add_argument(u"output", help=u"Path to the output file.", type=unicode)
    parser.add_argument(u"components", help=u"Inner dimension of the factorization.", type=int)
    parser.add_argument(u"epochs", help=u"Training epochs.", type=int)
    parser.add_argument(u"learning_rate", help=u"Learning rate to use.", type=float)
    parser.add_argument(u"decay_rate", help=u"Decay rate of the learning rate.", type=float)
    parser.add_argument(u"user_l2", help=u"L2 regularization on the user.", type=float)
    parser.add_argument(u"item_l2", help=u"L2 regularization on the item.", type=float)
    parser.add_argument(u"seed", help=u"Seed of the random generator.", type=int)
    args = parser.parse_args()
    main(args.input, args.output, args.components, args.epochs,
         args.learning_rate, args.decay_rate, args.user_l2, args.item_l2, args.seed)
	