"""Simple implementations of Matrix factorization with Gradient descent"""
import numpy as np

INVALID_FLOATS = [float("nan"), float("inf"), -float("inf")]
VALID_DTYPES = ["int", "float"]


class BasicAttr:
    """Basic Attributes of Matrix Factorization model"""
    def __init__(self, latent_size, lr, max_iter, tol):
        """Initialize Basic Attributes
            Args:
                latent_size (int): dimension of latent features
                lr (float): learning rate
                max_iter (int): maximum iteration of gradient descent
                tol (float): tolerance for early stopping
            Attr:
                user: user-latent matrix
                item: latent-item matrix
        """
        self.latents = latent_size
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self._size, self._user, self._item = None, None, None

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, inp):
        try:
            lr = float(inp)
            if lr in INVALID_FLOATS:
                raise ValueError
        except ValueError as err:
            raise err
        else:
            self._lr = lr

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, inp):
        try:
            max_iter = int(inp)
            if max_iter <= 0:
                raise ValueError
        except ValueError as err:
            raise err
        else:
            self._max_iter = max_iter

    @property
    def latents(self):
        return self._latents

    @latents.setter
    def latents(self, inp):
        try:
            latents = int(inp)
            if latents <= 0:
                raise ValueError
        except ValueError as err:
            raise err
        else:
            self._latents = latents

    @property
    def tol(self):
        return self._tol
    
    @tol.setter
    def tol(self, inp):
        try:
            tol = float(inp)
            if tol in INVALID_FLOATS:
                raise ValueError
        except ValueError as err:
            raise err
        else:
            self._tol = tol

    @property
    def user(self):
        return self._user

    @property
    def item(self):
        return self._item

    
class MF(BasicAttr):
    """A Simple MF implementation using Gradient Descent
        Attributes:
            latent_size: number of latent features
            lr: learning rate
            max_iter: maximum iteration of gradient descent
            tol: tolerance for early stopping
            size: size of input matrix, default=None
        Methods:
            loss(data): Returns MSE loss on given data
            predict(user, item): Make prediction of a particular item
                                 for a given user
            predict_user(user): Make predictions for a given user
            predict_all(): Make predictions on all users and items
            fit(data): Train model on given data

    """
    def __init__(self, latent_size, lr=0.02, max_iter=1000, tol=1e-5):
        """Initialize an object of GDMF
            Args:
                latent_size (int): number of latent features
                lr (float): learning rate
                max_iter (int): maximum iteration of gradient descent
                tol (float): tolerance for early stopping
        """
        super().__init__(latent_size, lr, max_iter, tol)

    def _init_matrix(self, mat):
        """Transform input matrix into np.ndarray and
           initialize attributes with input matrix,
           including user and item matrix and
           size of input matrix (self.size).
            Args:
                mat: input matrix
        """
        if not isinstance(mat, np.ndarray):
            try:
                mat = np.array(mat)
            except ValueError as err:
                raise err
        if mat.dtype not in VALID_DTYPES:
            raise ValueError("dtype should be int or float")

        user_shape = (mat.shape[0], self.latents)
        item_shape = (self.latents, mat.shape[1])
        self._user = np.random.uniform(
            low=0.1, high=0.9, size=user_shape)
        self._item = np.random.uniform(
            low=0.1, high=0.9, size=item_shape)

        self._size = mat.size

        return mat

    def __compute_mse(self, data, prediction):
        """Compute the MSE loss
            Args:
                data: targets
                prediction: predictions
            Returns:
                MSE of predictions
        """
        return np.sum((data - prediction)**2) / self._size

    def predict(self, user, item):
        """Make Prediction of a particular item for a given user
            Args:
                user (int): index of the user
                item (int): index of the item
            Returns:
                A prediction given by matrix production (float)
        """
        return np.dot(self.user[user, :], self.item[:, item])

    def predict_user(self, user):
        """Make Predictions of all items for a given user
            Args:
                user (int): index of the user
            Returns:
               Prediction (np.array(float))
        """
        return np.dot(self.user[user, :], self.item)

    def predict_all(self):
        """Make Prediction on all users and items
            Returns:
                Prediction (np.array(float))
        """
        return np.matmul(self.user, self.item)

    def _predict_known(self):
        """Make Prediction on all users and items,
           but not returning predictions for null data,
           this acted the same way as predict_all()
           for origin matrix factorization.
        """
        return self.predict_all()

    def loss(self, data):
        """Return MSE loss over a set of given data
            Args:
                data (np.ndarray): Matrix with the same size of
                                   training data
            Returns:
                mse_loss (float)
        """
        if self.user is None or self.item is None:
            return 0

        return self.__compute_mse(data, self._predict_known())

    def _get_grad_user(self, diff):
        """Function to compute gradient w.r.t. user matrix
            Args:
                diff (np.array): difference between predictions
                                 and targets
            Returns:
                gradient_user (np.array): gradient w.r.t. user
        """
        return np.matmul(diff, self.item.T)

    def _get_grad_item(self, diff):
        """Function to compute gradient w.r.t. item matrix
            Args:
                diff (np.array): difference between predictions
                                 and targets
            Returns:
                gradient_user (np.array): gradient w.r.t. item
        """
        return np.matmul(self.user.T, diff)

    def _update_gradient(self, data, pred):
        """Update gradients to user and item matrix
            Args:
                data (np.array): targets
                pred (np.array): predictions
        """
        diff = (pred - data) / self._size
        grad_user = self._get_grad_user(diff)
        grad_item = self._get_grad_item(diff)
        self._user -= self.lr * grad_user
        self._item -= self.lr * grad_item

    def fit(self, data):
        """Factorize target data into user and item matrix
            Args:
                data (np.ndarray): 2d array consists of targets
        """
        data = self._init_matrix(data)

        pred = self._predict_known()
        prev_loss = self.__compute_mse(data, pred)
        for epoch in range(self.max_iter):
            self._update_gradient(data, pred)
            pred = self._predict_known()
            loss = self.__compute_mse(data, pred)
            if abs(loss - prev_loss) < self.tol:
                print("stop at {} iteration".format(epoch+1))
                break
            prev_loss = loss


class MFWithNull(MF):
    """Matrix Factorization with Nulls
       This model treats 0 as null
       Attributes:
           maxn: desired upper-bond of prediction
           minn: desired lower-bond of prediction
           latents: number of latent features
           max_iter: maximum of iteration for gradient descent
           lr: learning rate
           tol: tolerance for early stopping
           __mask: indicate position of nonnulls
           __inv_mask: indicate position of nulls
    """
    def __init__(self, latent_size, maxn, minn, lr=0.02,
                 max_iter=1000, tol=1e-5):
        """Initialize an object of GDMFWithNull
            Args:
                latent_size (int): number of latent features
                maxn: desired upper-bond of prediction
                minn: desired lower-bond of prediction
                lr (float): learning rate, default=0.02
                max_iter (int): maximum iteration of gradient descent,
                                default=1000
                tol (float): tolerance for early stopping, default=1e-5
        """
        super().__init__(latent_size, lr, max_iter, tol)
        self.maxn = maxn
        self.minn = minn
        self.__mask = None
        self.__inv_mask = None

    @property
    def maxn(self):
        return self.__max

    @property
    def minn(self):
        return self.__min

    @maxn.setter
    def maxn(self, inp):
        if isinstance(inp, int) or isinstance(inp, float):
            self.__max = inp
            return
        raise ValueError("Max number should be integer or float")

    @minn.setter
    def minn(self, inp):
        if isinstance(inp, int) or not isinstance(inp, float):
            self.__min = inp
            return
        raise ValueError("Min number should be integer or float")
    
    def _init_matrix(self, mat):
        """Transform input matrix into np.ndarray and
           initialize attributes with input matrix,
           including user and item matrix, and masks.
            Args:
                mat: input matrix
        """
        if not isinstance(mat, np.ndarray):
            try:
                mat = np.array(mat)
            except ValueError as err:
                raise err
        if mat.dtype not in VALID_DTYPES:
            raise ValueError("dtype should be int or float")

        user_shape = (mat.shape[0], self.latents)
        item_shape = (self.latents, mat.shape[1])
        self._user = np.random.uniform(
            low=0.1, high=0.9, size=user_shape)
        self._item = np.random.uniform(
            low=0.1, high=0.9, size=item_shape)

        self.__mask = np.where(mat==0, 0, 1)
        self.__inv_mask = np.where(self.__mask==1, 0, 1)
        self._size = self.__mask.nonzero()[0].size

        return mat

    def _predict_known(self):
        """Only return predictions to nonnulls"""
        return self.predict_all() * self.__mask

    def _update_gradient(self, data, pred):
        """Update gradients to user and item matrix
            Args:
                data (np.array): targets
                pred (np.array): predictions
        """
        diff = self.__get_diff(data, pred)
        grad_user = self._get_grad_user(diff)
        grad_item = self._get_grad_item(diff)
        self._user -= self.lr * grad_user
        self._item -= self.lr * grad_item

    def __get_diff(self, data, pred):
        """Compute difference between origin data and prediction
           (When value is null(0), and prediction is out of range,
            fill null with minn if prediction < minn, or maxn if
            prediction > maxn)
            Args:
                data (np.ndarray): targets
                pred (np.ndarray): predictions
            Returns:
                diff (np.ndarray): difference / number of nonzero
        """
        adj_mat = pred * self.__inv_mask
        d_max = np.where(adj_mat > self.maxn, self.maxn, 0)
        d_min = np.where((0 != adj_mat) & (adj_mat < self.minn), self.minn, 0)
        p_max = np.where(adj_mat > self.maxn, adj_mat, 0)
        p_min = np.where(adj_mat < self.minn, adj_mat, 0)
        pred = pred * self.__mask + p_max + p_min
        diff = pred - d_max - d_min - data
        size = diff.nonzero()[0].size
        return diff / size