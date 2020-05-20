from __future__ import division
import numpy as np
import sciunit
import networkx as nx
from cv2 import EMD, DIST_L2

class wasserstein_distance(sciunit.Score):

    score = np.nan

    @classmethod
    def compute(self, observation, prediction, **kwargs):
        if observation.shape[0] == prediction.shape[0]:
            ndim = observation.shape[0]
        else:
            raise ValueError("Observation and prediction are not of the same "\
                           + "dimensionality!")

        disttype = DIST_L2
        N = observation.shape[1]  # number of observation neurons
        M = prediction.shape[1]  # number of prediciton neurons
        obsv_weights = np.ones(N, dtype=np.float32)
        pred_weights = np.ones(M, dtype=np.float32)

        observation_sig = np.append(obsv_weights[np.newaxis,:],
                                    observation.astype(np.float32), axis=0)
        prediction_sig  = np.append(pred_weights[np.newaxis,:],
                                    prediction.astype(np.float32), axis=0)

        ws_distance, _, _ = EMD(observation_sig.T, # array
                                prediction_sig.T, # array
                                distType=disttype, # int
                                # cost = noArray(), # array
                                # lowerBound = 0, # float
                                )

        self.score = wasserstein_distance(ws_distance)
        self.score.ndim = ndim
        self.score.obsv_samples = N
        self.score.pred_samples = M
        self.score.disttype = disttype
        return self.score


    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return f'{self.score}'
