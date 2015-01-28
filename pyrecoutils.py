import logging
import gzip
import sys
import numpy as np


class PredictionWriter:
	def __init__(self, filename):
		self.filename = filename
		self.writer = gzip.open(filename, u"wb")

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.writer.close()

	def space_line(self):
		self.writer.write(u"\n".encode(u"UTF-8", u"ignore"))

	def dump(self, reviews, predictions):
		for (user, item, rating), prediction in zip(reviews, predictions):
			self.writer.write((u"%s %s %+.1f %+.9f\n" % (user, item, rating, prediction)).encode(u"UTF-8", u"ignore"))


def rmse(reviews, predictions):
	return np.average([(r - p) ** 2 for (u, i, r), p in zip(reviews, predictions)])


def split_sets(dataset, training=0.8, validation=0.1):
	assert training + validation < 1., u"Invalid training and validation sizes %f and %f" % (training, validation)
	size = len(dataset)
	training_set = dataset[:int(training * size)]
	validation_set = dataset[int(training * size):int((training + validation) * size)]
	test_set = dataset[int((training + validation) * size):]
	logging.info(u"Split %d examples in %d, %d and %d", len(dataset), len(training_set), len(validation_set),
				 len(test_set))
	return training_set, validation_set, test_set


def load_ratings(data_filename):
	logging.info(u"Loading ratings from %s", data_filename)
	reviews = []
	with gzip.open(data_filename, u"rb") as data_file:
		for line in data_file:
			user, item, rating = line.decode(u"UTF-8", u"ignore").strip().split()[:3]
			reviews.append((user, item, float(rating)))
	logging.info(u"Loaded %d reviews from %s", len(reviews), data_filename)
	return reviews

			
if __name__ == u"__main__":
	logging.basicConfig(format=u'%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
	logging.info(u"Called: %s", u" ".join(sys.argv))
	reviews = load_ratings(sys.argv[1])
	logging.info(u"Number of users: %d", len(set([u for u, i, r in reviews])))
	logging.info(u"Number of items: %d", len(set([i for u, i, r in reviews])))
	logging.info(u"Average rating: %f", np.mean([r for u, i, r in reviews]))