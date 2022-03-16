from ldpc.protograph import array, identity, hstack, vstack, RingOfCirculantsF2

import numpy as np


pt_a = array([[(1)]])
pt_b = array([[(1), (2), (3)]])
pt_c = array([[(2), (4)], [(1), (3)]])
pt_d = array([[(2), (4)], [(1), (5)]])
pt_e = array([[(2), (3)]])
pt_f = array([[(2), (4), (2), (4)], [(1), (3), (1), (5)]])
pt_g = array([[(2)]])


def test_init():
    assert pt_a.shape == (1, 1)
    assert pt_b.shape == (1, 3)
    assert pt_c.shape == (2, 2)


def test_eq():
    assert pt_a[0] == RingOfCirculantsF2([1])
    assert (pt_b == array([[1, 2, 3]])).all()
    assert (pt_c[0] == array([[2, 4]])).all()
    assert array([[RingOfCirculantsF2([])]]) == array(
        [[RingOfCirculantsF2([])]])


def test_mul():
    assert(pt_e.T @ pt_a == array([[(-1)], [(-2)]])).all()


def test_T():
    assert (pt_a.T == array([[-1]])).all()
    assert (pt_b.T == array([[-1], [-2], [-3]])).all()


def test_hstack():
    assert (hstack([pt_a, pt_e]) == pt_b).all()
    assert (hstack([pt_c, pt_d]) == pt_f).all()


def test_kron():
    pt_e = array([[(2), (3)]])
    pt_b = array([[(1), (2), (3)]])
    assert(np.kron(pt_e, pt_b) == array([[3, 4, 5, 4, 5, 6]])).all()
    assert(np.kron(pt_e, pt_b).T == array(
        [[-3], [-4], [-5], [-4], [-5], [-6]])).all()


def test_vstack():
    assert (vstack([pt_a, pt_g]) == array([[1], [2]])).all()


def test_to_binary():
    assert (array.to_binary(pt_e, 2) == np.array(
        [[1, 0, 0, 1], [0, 1, 1, 0]])).all()


def test_identity():
    assert(identity(2) == array(
        [[(0), RingOfCirculantsF2([])], [RingOfCirculantsF2([]), (0)]])).all()


if __name__ == "__main__":

    a=array([[(1,2),0,1],[(),(4),(3)]])
    print(a.__compact_str__())
    
    import ldpc.protograph as pt

    a=pt.identity(30)
    b = pt.identity(30)

    c=np.kron(b,a)
    print(c.shape)
