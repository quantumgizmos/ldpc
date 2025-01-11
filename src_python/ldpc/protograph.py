import numpy as np
import copy as cp


def permutation_matrix(n: int, shift: int) -> np.ndarray:
    """
    Outputs a size-n permutation matrix.

    Parameters:
    -----------
        n: int
            matrix dimension
        shift: int
            the shift parameter

    Returns
    -------
        mat: nxn matrix shifted by `shift' columns to the left
    """
    return np.roll(np.identity(n), 1 * shift, axis=1).astype(int)


class RingOfCirculantsF2:
    """
    Class implementing the algebra of the ring of circulants over the field f2

    Parameters
    ----------
    non_zero_coefficients: int
        List of the non-zero terms in the polynomial expansion of the ring element
    """

    def __init__(self, non_zero_coefficients):
        try:
            self.coefficients = list(non_zero_coefficients)
        except TypeError:
            self.coefficients = [non_zero_coefficients]
        self.coefficients = np.array(self.coefficients).astype(int)
        try:
            assert len(self.coefficients.shape) == 1
        except AssertionError:
            raise TypeError(
                "The input to RingOfCirculantsF2 must be a one-dimensional list"
            )

        # coefficient simplification
        coefficients, counts = np.unique(self.coefficients, return_counts=True)
        self.coefficients = coefficients[counts % 2 == 1]

    def __type__(self):
        return RingOfCirculantsF2

    def __add__(self, other):
        """
        Overload for the addition operator between two ring elements.
        Removes duplicates from resulting ring element.

        Parameters
        ----------
        self: RingOfCirculantsF2
        other: RingOfCirculantsF2

        Returns
        -------
        RingOfCirculantsF2
        """
        return RingOfCirculantsF2(
            np.concatenate([self.coefficients, other.coefficients])
        )

    def __str__(self):
        """
        What we see when we print
        """
        return f"\u03bb{self.__repr__()}"

    def __repr__(self):
        """
        Re-usable output. Ie. Is valid code
        """
        length = self.len()
        out = "("
        for i, value in enumerate(self.coefficients):
            out += str(value)
            if i != (length - 1):
                out += ","
        out += ")"
        return out

    def __eq__(self, other):
        if type(other) == RingOfCirculantsF2:
            if self.coefficients.shape != other.coefficients.shape:
                return False

            else:
                if sorted(self.coefficients) != sorted(other.coefficients):
                    return False
                return True
        elif other is None:
            return False
        else:
            if len(self.coefficients) == len(other):
                return (self.coefficients == other).all()
            return False

    @property
    def T(self):
        """
        Returns the transpose of an element from the ring of circulants

        Returns
        -------
        RingOfCirculantsF2
        """
        transpose_coefficients = -1 * self.coefficients
        return RingOfCirculantsF2(transpose_coefficients)

    def __mul__(self, other):
        """
        Overloads the multiplication operator * between elements of the ring of circulants
        """

        if isinstance(other, (int, float)):
            return self.__rmul__(other)

        try:
            assert type(self) == type(other)
        except AssertionError:
            raise TypeError(
                f"Ring elements can only be multiplied by other ring elements. Not by {type(other)}"
            )

        no_coeffs = self.len() * other.len()
        new_coefficients = np.zeros(no_coeffs).astype(int)
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                new_coefficients[i * other.len() + j] = a + b

        return RingOfCirculantsF2(new_coefficients)

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            if int(other) % 2 == 0:
                return RingOfCirculantsF2(())
            else:
                return self

    def len(self):
        return len(self.coefficients)

    def __len__(self):
        return len(self.coefficients)

    def to_binary(self, lift_parameter):
        """
        Converts ring element to its binary representation

        Parameters
        ----------
        lift_parameter:int
            The size of the permutation matrices used to map to binary

        Returns
        numpy.ndarray
            Binary matrix in numpy format
        """
        mat = np.zeros((lift_parameter, lift_parameter)).astype(int)
        for coeff in self.coefficients:
            mat += permutation_matrix(lift_parameter, coeff)
        return mat % 2


class array(np.ndarray):
    """
    Class implementing a protograph (an array where the elements are in the ring of circulants)


    Parameters
    ----------
    proto_array: array_like, 2D
        The input should be of the form [[(0,1),(),(1)]] where each tuple is the input to the RingOfCirculantsF2 class
    """

    def __new__(cls, proto_array):
        # Reads in input arrays and converts tuples to RingOfCirculantsF2 objects
        temp_proto = np.array(proto_array).astype(object)
        if len(temp_proto.shape) == 3:
            m, n, _ = temp_proto.shape
        elif len(temp_proto.shape) == 2:
            m, n = temp_proto.shape
        else:
            raise TypeError(
                "The input protograph must be a three-dimensional array like object or a two-dimensional array with elements that are tuples"
            )
        proto_array = np.array(
            [
                temp_proto[i, j]
                if isinstance(temp_proto[i, j], RingOfCirculantsF2)
                else RingOfCirculantsF2(temp_proto[i, j])
                for i in range(m)
                for j in range(n)
            ]
        )
        proto_array.shape = (m, n)
        return proto_array.view(cls)

    @property
    def T(self):
        """
        Returns the transpose of the protograph
        """
        m, n = self.shape
        temp = np.copy(self)
        for i in range(m):
            for j in range(n):
                temp[i, j] = temp[i, j].T

        return temp.T.view(type(self))

    def to_binary(self, lift_parameter):
        """
        Converts the protograph to binary
        """
        L = lift_parameter
        m, n = self.shape
        mat = np.zeros((m * L, n * L)).astype(int)
        for i in range(m):
            for j in range(n):
                mat[i * L : (i + 1) * L, j * L : (j + 1) * L] = self[i, j].to_binary(L)
        return mat

    @property
    def copy(a):
        """
        Copies a protograph
        """
        return cp.deepcopy(a)

    def __str__(self):
        """
        Generates what we see when we print
        """

        m, n = self.shape
        out = "[["

        for i in range(m):
            if i != 0:
                out += " ["
            for j in range(n):
                out += str(self[i, j])
                if j != n - 1:
                    out += " "
            if i != m - 1:
                out += "]\n"
            else:
                out += "]]"

        return out

    def __compact_str__(self):
        """
        Generates what we see when we print
        """

        m, n = self.shape
        out = "[["

        for i in range(m):
            if i != 0:
                out += " ["
            for j in range(n):
                out += repr(self[i, j])
                if j != n - 1:
                    out += " "
            if i != m - 1:
                out += "]\n"
            else:
                out += "]]"

        return out


def identity(size):
    """
    Returns an identity protograph
    """
    proto = zeros(size)
    for j in range(size):
        proto[j, j] = RingOfCirculantsF2([0])
    return proto


def zeros(size):
    """
    Returns a protograph full of zero elements from the ring of circulants
    """
    if isinstance(size, int):
        m = size
        n = size
    else:
        m = size[0]
        n = size[1]

    proto_array = np.zeros((m, n)).astype(object)
    for i in range(m):
        for j in range(n):
            proto_array[i, j] = RingOfCirculantsF2([])
    return array(proto_array)


def hstack(proto_list):
    """
    hstack funciton for protographs
    """
    return np.hstack(proto_list).view(array)


def vstack(proto_list):
    """
    vstack function for protographs
    """
    return np.vstack(proto_list).view(array)
