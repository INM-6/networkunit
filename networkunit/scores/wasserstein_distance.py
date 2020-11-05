from __future__ import division
import numpy as np
import sciunit
from scipy.stats import zscore
import networkx as nx
from cv2 import EMD, DIST_L2


class wasserstein_distance(sciunit.Score):
    score = np.nan
    _best = 0.
    _worst = np.nan_to_num(np.inf)

    @classmethod
    def compute(self, observation, prediction, **kwargs):
        if observation.shape[0] == prediction.shape[0]:
            ndim = observation.shape[0]
        else:
            raise ValueError("Observation and prediction are not of the same "
                             + "dimensionality!")

        # Filter NaNs
        observation_mask = np.all(np.isfinite(observation), axis=0)
        observation = observation[:, observation_mask]
        prediction_mask = np.all(np.isfinite(prediction), axis=0)
        prediction = prediction[:, prediction_mask]

        disttype = DIST_L2
        N = observation.shape[1]  # number of observation neurons
        M = prediction.shape[1]  # number of prediciton neurons

        if N and M:
            # Normalize
            obsv_pred = np.concatenate((observation, prediction), axis=1)
            obsv_pred = zscore(obsv_pred, axis=1, nan_policy='omit')
            observation, prediction = obsv_pred[:, :N], obsv_pred[:, N:]

            obsv_weights = M * np.ones(N, dtype=np.float32)
            pred_weights = N * np.ones(M, dtype=np.float32)

            observation_sig = np.append(obsv_weights[np.newaxis, :],
                                        observation.astype(np.float32), axis=0)
            prediction_sig = np.append(pred_weights[np.newaxis, :],
                                       prediction.astype(np.float32), axis=0)

            ws_distance, _, _ = EMD(observation_sig.T,  # array
                                    prediction_sig.T,  # array
                                    distType=disttype,  # int
                                    # cost = noArray(), # array
                                    # lowerBound = 0, # float
                                    )
        else:
            print("Warning (scores.wasserstein_distance): ",
                  "Predictions or observations are empty! ",
                  "Returning maximum value.")
            ws_distance = self._worst

        self.score = wasserstein_distance(ws_distance)
        self.score.ndim = ndim
        self.score.obsv_samples = N
        self.score.pred_samples = M
        self.score.obsv_silent_ratio = np.sum(~observation_mask)/len(observation_mask)
        self.score.pred_silent_ratio = np.sum(~prediction_mask)/len(prediction_mask)
        self.score.disttype = disttype
        return self.score

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return f'{self.score}'
