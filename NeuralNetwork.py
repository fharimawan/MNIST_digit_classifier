import numpy as np
import random
from psutil._compat import xrange

class NeuralNetwork:
    def __init__(self, sizes):
        self.nlayers = len(sizes)
        self.sizes = sizes
        # initialize weights and biases randomly with appropriate sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w.dot(a) + b)
        return a

    def train(self, data, epochs, mini_batch_size, eta, test_data=None, with_cost=False):
        '''Trains NN using stochastic gradient descent'''
        n = len(data)
        if test_data:
            num_tests = len(test_data)

        for e in xrange(epochs):
            # split data into mini batches
            np.random.shuffle(data)
            mini_batches = [data[batch: batch + mini_batch_size] for batch in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                correct, cost = self.evaluate(test_data, with_cost)
                print(f'Epoch {e+1}: {correct} / {num_tests}')
                if with_cost:
                    print(f'          {cost}')
            else:
                print(f'Epoch {e+1} completed')
        

    def update_mini_batch(self, mini_batch, eta):
        # init bias and weight gradients with zero vector in appropriate shape
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # for each datum compute the gradient and sum it
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_prop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update weights and biases by making a gradient step in the direction of the summed gradients
        self.biases = [b - nb * (eta / len(mini_batch)) for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - nw * (eta / len(mini_batch)) for w, nw in zip(self.weights, nabla_w)]

    def back_prop(self, x, y):
        ''' x is a np.array of the feature values
            y is the label for the data with these features'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # compute forward pass and store the responses (z) and activations AKA logistic responses (a)
        a = x
        activations = [x]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = w.dot(a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        # compute backwards pass
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(activations[-2].T)
        for l in range(2, self.nlayers):
            delta = self.weights[-l+1].T.dot(delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = delta.dot(activations[-l-1].T)
        return nabla_b, nabla_w

    def evaluate(self, test_data, with_cost=False):      
        res = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        cost = None
        if with_cost:
            cost = 0
            for yhat, y in res:
                cost += (y - yhat)**2
        return sum(int(x == y) for (x, y) in res), cost / (2*len(test_data))

    def predict(self, xs):
        res = [(np.argmax(self.feedforward(x)), y) for x, y in xs]
        return res
            

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1. - sigmoid(z))