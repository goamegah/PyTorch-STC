import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


def random_init_clustering(n_clusters, n_rows, random_state=None):
    """Create a random row cluster assignment matrix.

    Each row contains 1 in the column corresponding to the cluster where the
    processed data matrix row belongs, 0 elsewhere.

    Parameters
    ----------
    n_clusters: int
        Number of clusters
    n_rows: int
        Number of rows of the data matrix (i.e. also the number of rows of the
        matrix returned by this function)
    random_state : int or :class:`numpy.RandomState`, optional
        The generator used to initialize the cluster labels. Defaults to the
        global numpy random number generator.

    Returns
    -------
    matrix
        Matrix of shape (``n_rows``, ``n_clusters``)
    """

    random_state = check_random_state(random_state)
    Z_a = random_state.randint(n_clusters, size=n_rows)
    Z = np.zeros((n_rows, n_clusters))
    Z[np.arange(n_rows), Z_a] = 1
    return Z


def check_array(a, pos=True):
    """Check if an array contains numeric values with non empty rows nor
    columns.

    Parameters
    ----------
    a:
        The input array
    pos: bool
        If ``True``, check if the values are positives

    Raises
    ------
    TypeError
        If the array is not a Numpy/SciPy array or matrix or if the values are
        not numeric.

    ValueError
        If the array contains empty rows or columns or contains NaN values, or
        negative values (if ``pos`` is ``True``).
    """

    if not (type(a) == np.ndarray or type(a) == np.matrix or sp.issparse(a)):
        raise TypeError("Input data must be a Numpy/SciPy array or matrix")

    if (not np.issubdtype(a.dtype.type, np.integer) and
            not np.issubdtype(a.dtype.type, np.floating)):
        raise TypeError("Input array or matrix must be of a numeric type")

    if not sp.issparse(a):
        a = np.matrix(a)

        if len(np.where(~a.any(axis=0))[0]) > 0:
            raise ValueError("Zero-valued columns in data")
        if len(np.where(~a.any(axis=1))[1]) > 0:
            raise ValueError("Zero-valued rows in data")
        if pos:
            if (a < 0).any():
                raise ValueError("Negative values in data")
        if np.isnan(a).any():
            raise ValueError("NaN in data")


class SphericalKmeans:
    """Spherical k-means clustering.

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Number of clusters to form
    init : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial column labels
    max_iter : int, optional, default: 20
        Maximum number of iterations
    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    tol : float, default: 1e-9
        Relative tolerance with regards to criterion to declare convergence
    weighting : boolean, default: True
        Flag to activate or deactivate TF-IDF weighting

    Attributes
    ----------
    labels_ : array-like, shape (n_rows,)
        cluster label of each row
    criterion : float
        criterion obtained from the best run
    criterions : list of floats
        sequence of criterion values during the best run
    """

    def __init__(self, n_clusters=2, init=None, max_iter=20, n_init=1,
                 tol=1e-9, random_state=None, weighting=True):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.labels_ = None
        self.criterions = []
        self.criterion = -np.inf
        self.weighting = weighting
        self.Z = None
        self.Z_fuzzy = None
        self.centers = None

    def fit(self, X, y=None):
        """Perform clustering.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """

        check_array(X, pos=False)

        if X.shape[0] < self.n_clusters:
            raise ValueError("data matrix has not enough rows")

        criterion = self.criterion

        if self.weighting:
            transformer = TfidfTransformer(norm='l2', smooth_idf=True)
            X = transformer.fit_transform(X)

        X = X.todense()
        X = np.array(X)
        X = sp.lil_matrix(X)
        #X = sp.csr_matrix(X)
        X = normalize(X)

        #X = X.astype(float)

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            # print(" == New init == ")
            self.random_state = seed
            self._fit_single(X)
            # remember attributes corresponding to the best criterion
            if (self.criterion > criterion):
                criterion = self.criterion
                criterions = self.criterions
                labels_ = self.labels_
                z = self.Z
                z_fuzzy = self.Z_fuzzy

        self.random_state = random_state

        # update attributes
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = labels_
        self.Z = z
        self.Z_fuzzy = z_fuzzy

    def _fit_single(self, X, y=None):
        """Perform one run of clustering.

        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        K = self.n_clusters

        if self.init is None:
            Z = random_init_clustering(K, X.shape[0], self.random_state)
        else:
            Z = np.matrix(self.init, dtype=float)

        X = sp.lil_matrix(X)

        Z = sp.lil_matrix(Z)  # random_init function returns a nd_array

        change = True

        c_init = -np.inf
        c_list = []
        n_iter = 0

        while change and n_iter < self.max_iter:
            print("iteration:", n_iter)
            change = False

            # compute centroids (in fact only summation along cols)
            centers = Z.T*X  # centers = sparse matrix

            # normalize centroids
            centers = normalize(centers)

            # hard assignment
            #Z=centers*X.T
            Z1 = X * centers.T
            Z1 = Z1.todense()
            Z1 = np.array(Z1)
            Z = np.zeros_like(Z1)
            Z[np.arange(len(Z1)), Z1.argmax(1)] = 1
            Z = sp.csc_matrix(Z)

            # compute and check if cosine criterion still evolves
            k_times_k = Z.T * X * centers.T
            c = np.trace(k_times_k.todense())  # no trace for sp ...

            if np.abs(c - c_init) > 1e-9:
                c_init = c
                change = True
                c_list.append(c)
                # print(c)
            n_iter += 1

        self.criterion = c
        self.criterions = c_list
        part = Z.todense().argmax(axis=1).tolist()
        self.labels_ = [item for sublist in part for item in sublist]
        self.Z = Z
        self.Z_fuzzy = Z1
        self.centers = centers.toarray()
