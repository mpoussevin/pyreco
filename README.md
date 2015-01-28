# pyreco
Simple recommender systems in python.
A collection of scripts to learn and output the prediction of simple recommender systems.
They share a similar behavior: 
* The input file is expected to be a gunzip text file with at list 3 columns, separated by spaces: USER_ID ITEM_ID RATING
* The input file is automatically split in 3 (training, validation and test) using 80%, 10% and 10% of the data.
* The output file is then a gunzip text file with 4 columns: USER_ID ITEM_ID RATING PREDICTION
* The training, validation and test sets are separated by empty lines in the output file.

## Bias models
* overall_bias.py: overall bias model (average rating). Takes two parameters:
  * input: the path to the input file (one file for training, validation and test)
  * output: the path to the output file (one file for training, validation and test) where predictions are written
* user_bias.py: user bias model (average rating per user). Takes two parameters:
  * input: the path to the input file (one file for training, validation and test)
  * output: the path to the output file (one file for training, validation and test) where predictions are written
* item_bias.py: item bias model (average rating per item). Takes two parameters:
  * input: the path to the input file (one file for training, validation and test)
  * output: the path to the output file (one file for training, validation and test) where predictions are written

## Collaborative filtering model
* matfact.py: uses adaptive biases and collaborative filtering. It uses a stochastic gradient descent to solve the rating prediction problem as in the work of Yehuda Koren, Robert Bell and Chris Volinsky in 2009 (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.147.8295). Takes multiple parameters:Takes two parameters:
  * input: the path to the input file (one file for training, validation and test)
  * output: the path to the output file (one file for training, validation and test) where predictions are written
  * components: number of components to use for the factorisation (dimension of the latent space)
  * epochs: number of epochs to user for training (start with 10 to 50)
  * learning_rate: initial value of the learning rate (start with 1e-2 or 1e-3)
  * decay_rate: the decay of the learning rate at each epoch (lr *= decay, start with 0.99)
  * user_l2: the weight of the L2 regularization for users (start with 1e-2)
  * item_l2: the weight of the L2 regularization for items (start with 1e-2)
  * seed: the seed of the random generator (start with 42)
 
