"""
The Singular Angle Similarity (SAS) Score is introduced in Albers, Kurth et al. (2024)
doi: doi.org/10.5281/zenodo.10680478
https://github.com/INM-6/SAS
"""

import numpy as np
import sciunit


class singularangle(sciunit.Score):
    """
    The singular angle score evaluates whether two real matrices have similar
    structure by measuring the angles between the corresponding singular vectors
    and weighing them with their singular values.
    """

    score = np.nan

    @classmethod
    def compute(self, matrix_a, matrix_b):

        U_a, S_a, V_at = np.linalg.svd(matrix_a)
        U_b, S_b, V_bt = np.linalg.svd(matrix_b)

        # if the matrices are rectangular, disregard the singular vectors
        # of the larger singular matrix that map to 0
        dim_0, dim_1 = matrix_a.shape
        if dim_0 < dim_1:
            V_at = V_at[:dim_0, :]
            V_bt = V_bt[:dim_0, :]
        elif dim_0 > dim_1:
            U_a = U_a[:, :dim_1]
            U_b = U_b[:, :dim_1]

        U_angle = self._angle(U_a, U_b, method="columns")
        V_angle = self._angle(V_at, V_bt, method="rows")

        angles_noflip = (U_angle + V_angle) / 2
        angles_flip = np.pi - angles_noflip
        angles = np.minimum(angles_noflip, angles_flip)
        weights = (S_a + S_b) / 2

        # if one singular vector projects to 0, discard it
        zero_mask = (S_a > np.finfo(float).eps) | (S_b > np.finfo(float).eps)
        weights = weights[zero_mask]
        angles = angles[zero_mask]

        weights /= np.sum(weights)
        smallness = 1 - angles / (np.pi / 2)
        weighted_smallness = smallness * weights
        similarity_score = np.sum(weighted_smallness)

        self.score = singularangle(similarity_score)
        self.score.data_size = (dim_0, dim_1)
        return self.score

    def _angle(self, a, b, method="columns"):
        """
        Calculates the angles between the row or column vectors of
        two matrices.

        Parameters
        ----------
        a : ndarray
            First input matrix.
        b : ndarray
            Second input matrix.
        method : str, optional
            Defines the direction of the vectors (either 'rows' or 'columns'),
            by default 'columns'.

        Returns
        -------
        ndarray
            Array of angles.
        """
        if method == "columns":
            axis = 0
        if method == "rows":
            axis = 1

        dot_product = np.sum(a * b, axis=axis)
        magnitude_a = np.linalg.norm(a, axis=axis)
        magnitude_b = np.linalg.norm(b, axis=axis)
        angle = np.arccos(dot_product / (magnitude_a * magnitude_b))

        mask_pos1 = np.isnan(angle) & np.isclose(dot_product, 1)
        angle[mask_pos1] = 0
        mask_neg1 = np.isnan(angle) & np.isclose(dot_product, -1)
        angle[mask_neg1] = np.pi

        return angle

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return (
            "\n\n\033[4mSingular Angle Score\033[0m"
            + "\n\tdatasize: {} x {}".format(
                self.data_size[0], self.data_size[1]
            )
            + "\n\tscore = {:.3f}".format(self.score)
        )
