import os
import numpy as np
from maskloader import *


class MaskSVD(object):
    """
    a model intended for sparse matrix factorization with unknown elements masked
    either representation can be used as a possible embedding of row/column variable

    attributes:
        x: interaction matrix
        mask: characteristic matrix of unknown elements
        repr: two low-rank representation matrix u and i
        params:
            number of rows Nu,
            number of columns Ni,
            dimensions of represention repr_dim
            numIter: number of iterations
            lr: learning rate
            c: tuple of regularization coefficients (c1,c2)
    """

    def __init__(self, x, mask, y=None, unmask=None, repr_tuple=None, kind="mse", repr_dim=10, lr=0.01, c=(0, 0),
                 numIter=2000, verbose=True, evaluation=True):
        """
        initialize a model
        inputs:
            x: matrix of interaction
            repr_tuple: tuple of representation matrices (u,i)
            repr_dim: dimensions of representation
        """
        Nu, Ni = x.shape
        self.x = x
        self.mask = mask
        self.y = y
        self.unmask = unmask
        # initial the representation dict
        self.params = {}
        self.params["Nu"] = Nu
        self.params["Ni"] = Ni
        self.params["repr_dim"] = repr_dim
        self.params["numIter"] = numIter
        self.params["lr"] = lr
        self.params["kind"] = kind
        self.params["c"] = c
        self.verbose = verbose
        self.evaluation = evaluation

        if repr_tuple == None:
            self.repr = {}
            self.repr["u"] = .01 * np.random.randn(Nu, repr_dim)
            self.repr["i"] = .01 * np.random.randn(repr_dim, Ni)
        else:
            assert len(repr_tuple) == 2 and all(list(map(lambda x: type(x) == np.ndarray, repr_tuple)))
            self.repr["u"], self.repr["i"] = repr_tuple

    def loss(self, xhat, kind="mse"):
        """
        compute mean square error or binary cross entropy
        return:
            loss: mse/bce loss + L2 regularization(optional)
            grad: gradient for backpropagation
        """
        c1, c2 = self.params["c"]
        obs = 1 - self.mask
        nobs = len(obs)
        L2_regul = c1 * (np.sum(np.power(self.repr["u"], 2))) + c2 * (np.sum(np.power(self.repr["i"], 2)))
        if kind == "mse":
            loss = np.sum(np.power((self.x - xhat) * obs, 2) * obs) / nobs
            grad = -2 * (self.x - xhat) * obs / nobs
        elif kind == "bce":
            prob = 1 / (1 + np.exp(-xhat))
            loss = -np.sum((self.x * np.log(prob) + (1 - self.x) * np.log(1 - prob)) * obs) / nobs
            grad = (-self.x + prob) * obs / nobs
        else:
            raise ValueError("no such kind of loss")
        return {"loss": loss + L2_regul, "grad": grad}

    def multiplication(self):
        return self.repr["u"] @ self.repr["i"]

    def update(self, losses, lr):
        grad = losses["grad"]
        du = losses["grad"] @ self.repr["i"].T
        di = self.repr["u"].T @ losses["grad"]
        c1, c2 = self.params["c"]
        self.repr["u"] += -lr * (du + c1 * self.repr["u"])
        self.repr["i"] += -lr * (di + c2 * self.repr["i"])

    def train(self):
        numIter, lr = self.params["numIter"], self.params["lr"]
        kind = self.params["kind"]
        verbose = self.verbose
        evaluation = self.evaluation
        for k in range(numIter):
            xhat = self.multiplication()
            losses = self.loss(xhat, kind)
            if verbose:
                if k == 0 or (k + 1) % 100 == 0:
                    print("{}/{} Iterations -- Loss: {}".format(k + 1, numIter, losses["loss"]))
                    self.evaluate()
                    if evaluation:
                        if self.y is None or self.unmask is None:
                            raise ValueError("missing arguments y and unmask")
                        else:
                            self.evaluate(self.y, self.unmask)
            self.update(losses, lr)

    def evaluate(self, y=None, unmask=None):
        """
        evaluate the model through unmasking true label
        inputs:
            y: true label of like or dislike w.r.t. the target user
            unmask: the entries to be evaluate
        """
        eval_type = "Test"
        if y is None or unmask is None:
            y, unmask = self.x[-1] >= 0, (1 - self.mask)[-1]
            eval_type = "Train"
        obs_indices = np.nonzero(unmask)[0]
        y_pred = self.multiplication()[-1][obs_indices] >= 0
        y_true = y[obs_indices]
        accuracy = np.sum(np.equal(y_pred, y_true)) / len(obs_indices)
        print(eval_type + " Accuracy: {}".format(accuracy))

def maskevaluator(x,y,mask,unmask,cache=None):
    """
    inputs:
        x: reduced rating matrix containing target user and k nearest neighbors
           note that target user is always on the last row
        y: true label of like or dislike w.r.t. the target user
        mask: the entries unobserved in x
        unmask: the entries to be evaluate
    """
    model = MaskSVD(x_train,mask)
    model.train()
    obs_indices = np.nonzero(unmask)[0]
    y_pred = model.multiplication()[-1][obs_indices] >= 0
    y_true = y[obs_indices]
    accuracy = np.sum(np.equal(y_pred,y_true))/len(obs_indices)
    print("Model Accuracy: {}.".format(accuracy))

if __name__ == "__main__":
    # take user 0 as an example
    cache = loader("processed\\ratings_train.npz", "processed\\ratings_test.npz")
    x_train, y_test, mask, unmask = maskloader(0, cache=cache)
    maskevaluator(x_train, y_test[-1], mask, unmask[-1], cache=cache)