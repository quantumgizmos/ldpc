from ldpc.protograph import RingOfCirculantsF2

a = RingOfCirculantsF2([1, 3])
b = RingOfCirculantsF2([3, 1])
c = RingOfCirculantsF2([2, 3])
d = RingOfCirculantsF2([1])
e = RingOfCirculantsF2([])
f = RingOfCirculantsF2([1,1])


def test_init_RingOfCirculants():
    assert d.coefficients == 1


def test_equality():
    assert (a == b) == True
    assert (a == c) == False
    assert a == [1, 3]
    assert d == [1]


def test_addition_RingOfCirculants():
    assert a+b == e
    assert (a+c) == [1, 2]
    assert (d+c) != [1, 2]
    assert (d+d+d) == [1]
    assert (e+e) == []
    assert (a+a) == []
    assert (b+b) == []
    assert (f+f) == []
    assert (f+f) == e
    assert (f+d) == [1]

    # assert (f+d) == [1]


def test_len():
    assert len(a) == 2

if __name__ == "__main__":
    test_addition_RingOfCirculants()

    a=RingOfCirculantsF2([0,1,1,1,1,2,3,3,3,-3,-1])
    print(a)



