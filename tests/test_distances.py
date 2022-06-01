import numpy as np
from ncfs import distances
import unittest


class ManhattanTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.w_ = np.ones(3)

    def test_manhattan_x1_x2(self):
        dist = distances.manhattan(self.X[:, 0], self.X[:, 1], self.w_)
        self.assertAlmostEqual(dist, 3)

    def test_manhattan_x1_x3(self):
        dist = distances.manhattan(self.X[:, 0], self.X[:, 2], self.w_)
        self.assertAlmostEqual(dist, 6)

    def test_manhattan_x1_x1(self):
        dist = distances.manhattan(self.X[:, 0], self.X[:, 0], self.w_)
        self.assertAlmostEqual(dist, 0)

    def test_manhattan_pdist(self):
        X_dist = np.array([[0, 9, 18], [9, 0, 9], [18, 9, 0]])
        dist = np.zeros((3, 3))
        distances.pdist(self.X, self.w_, dist, distances.manhattan, symmetric=False)
        np.testing.assert_allclose(X_dist, dist)

    def test_manhattan_partials0(self):
        partial = distances.Manhattan(self.X, self.w_)
        expected = np.array([[0, 6, 12], [6, 0, 6], [12, 6, 0]])
        D = np.zeros((3, 3))
        partial.partials(self.X, D, 0)
        np.testing.assert_allclose(expected, D)

    def test_manhattan_partials1(self):
        partial = distances.Manhattan(self.X, self.w_)
        expected = np.array([[0, 6, 12], [6, 0, 6], [12, 6, 0]])
        D = np.zeros((3, 3))
        partial.partials(self.X, D, 1)
        np.testing.assert_allclose(expected, D)

    def test_manhattan_partials2(self):
        partial = distances.Manhattan(self.X, self.w_)
        expected = np.array([[0, 6, 12], [6, 0, 6], [12, 6, 0]])
        D = np.zeros((3, 3))
        partial.partials(self.X, D, 2)
        np.testing.assert_allclose(expected, D)

    def test_pdist_init(self):
        dist = np.zeros((3, 3))
        distances.pdist(self.X, self.w_, dist, distances.manhattan, symmetric=False)
        dist2 = np.ones((3, 3))
        np.fill_diagonal(dist2, 0.0)
        distances.pdist(self.X, self.w_, dist2, distances.manhattan, symmetric=False)
        np.testing.assert_equal(dist, dist2)

    def test_pdist_twice(self):
        dist = np.zeros((3, 3))
        distances.pdist(self.X, self.w_, dist, distances.manhattan, symmetric=False)
        first_dist = dist.copy()
        distances.pdist(self.X, self.w_, dist, distances.manhattan, symmetric=False)
        np.testing.assert_equal(first_dist, dist)


class SqeuclideanTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.w_ = np.ones(3)

    def test_sqeuclidean_x1_x2(self):
        dist = distances.sqeuclidean(self.X[:, 0], self.X[:, 1], self.w_)
        self.assertAlmostEqual(dist, 3)

    def test_sqeuclidean_x1_x3(self):
        dist = distances.sqeuclidean(self.X[:, 0], self.X[:, 2], self.w_)
        self.assertAlmostEqual(dist, 12)

    def test_sqeuclidean_x1_x1(self):
        dist = distances.sqeuclidean(self.X[:, 0], self.X[:, 0], self.w_)
        self.assertAlmostEqual(dist, 0)

    def test_sqeuclidean_pdist(self):
        X_dist = np.array([[0, 27, 108], [27, 0, 27], [108, 27, 0]])
        dist = np.zeros((3, 3))
        distances.pdist(self.X, self.w_, dist, distances.sqeuclidean, symmetric=False)
        np.testing.assert_allclose(X_dist, dist)

    def test_sqeuclidean_partials0(self):
        partial = distances.SqEuclidean(self.X, self.w_)
        expected = np.array([[0, 18, 72], [18, 0, 18], [72, 18, 0]])
        D = np.zeros((3, 3))
        partial.partials(self.X, D, 0)
        np.testing.assert_allclose(expected, D)


class EuclideanTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.w_ = np.ones(3)

    def test_euclidean_x1_x2(self):
        dist = distances.euclidean(self.X[:, 0], self.X[:, 1], self.w_)
        self.assertAlmostEqual(dist, np.sqrt(3))

    def test_euclidean_x2_x3(self):
        dist = distances.euclidean(self.X[:, 0], self.X[:, 2], self.w_)
        self.assertAlmostEqual(dist, np.sqrt(12))

    def test_euclidean_x1_x1(self):
        dist = distances.euclidean(self.X[:, 0], self.X[:, 0], self.w_)
        self.assertAlmostEqual(dist, 0)

    def test_euclidean_pdist(self):
        X_dist = np.array(
            [
                [0, 5.196152422706632, 10.392304845413264],
                [5.196152422706632, 0, 5.196152422706632],
                [10.392304845413264, 5.196152422706632, 0],
            ]
        )
        dist = np.zeros((3, 3))
        distances.pdist(self.X, self.w_, dist, distances.euclidean, symmetric=False)
        np.testing.assert_allclose(X_dist, dist)

    def test_sqeuclidean_partials0(self):
        partial = distances.Euclidean(self.X, self.w_)
        val1 = 9 / 5.196152422706632
        val2 = 36 / 10.392304845413264
        expected = np.array([[0, val1, val2], [val1, 0, val1], [val2, val1, 0]])

        D = np.zeros((3, 3))
        partial.partials(self.X, D, 0)
        np.testing.assert_allclose(expected, D)


class VarianceTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.w_ = np.ones(3)

    def test_variance(self):
        var = distances.variance(self.X[:, 0], self.w_)
        self.assertAlmostEqual(var, 9)

    def test_variance_zero_weights(self):
        var = distances.variance(self.X[:, 0], np.zeros(3))
        self.assertTrue(np.isnan(var), msg="{} not nan.".format(var))

    def test_variance_zero_feature(self):
        var = distances.variance(np.zeros(3), self.w_)
        self.assertAlmostEqual(var, 0)

    def test_variance_zero_sum_of_weights(self):
        var = distances.variance(self.X[:, 0], np.array([-1, 0, 1]))
        self.assertTrue(np.isnan(var), msg="{} not nan.".format(var))


class PhiSTest(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3])
        self.y = np.array([4, 2, 0])
        self.A = np.vstack((self.x, self.y, np.array([1, 1, 1])))
        self.w_ = np.ones(3)

    def test_xy(self):
        dist = distances.phi_s(self.x, self.y, self.w_)
        self.assertAlmostEqual(dist, 9)

    def test_yx(self):
        dist = distances.phi_s(self.y, self.x, self.w_)
        self.assertAlmostEqual(dist, 9)

    def test_2x2y(self):
        dist = distances.phi_s(2 * self.x, 2 * self.y, self.w_)
        self.assertAlmostEqual(dist, 9)

    def test_2y2x(self):
        dist = distances.phi_s(2 * self.y, 2 * self.x, self.w_)
        self.assertAlmostEqual(dist, 9)

    def test_phis_zero_weights(self):
        dist = distances.phi_s(self.x, self.y, np.zeros(3))
        self.assertTrue(np.isnan(dist), msg="{} not nan.".format(dist))

    def test_pdist(self):
        A_dist = np.array([[0, 9, 1], [9, 0, 1], [1, 1, 0]])
        dist = np.zeros((3, 3))
        distances.pdist(self.A, self.w_, dist, distances.phi_s, symmetric=True)
        np.testing.assert_allclose(A_dist, dist)

    def test_pdist_not_symmetric(self):
        A_dist = np.array([[0, 9, 1], [9, 0, 1], [1, 1, 0]])
        dist = np.zeros((3, 3))
        distances.pdist(self.A, self.w_, dist, distances.phi_s, symmetric=False)
        np.testing.assert_allclose(A_dist, dist)

    # def test_partials(self)

    # def test_update values


if __name__ == "__main__":
    #%%
    unittest.main(verbosity=3)
    suite = unittest.TestLoader().discover(".")
    unittest.TextTestRunner(verbosity=1).run(suite)
