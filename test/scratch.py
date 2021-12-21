import numpy as np

def test(c,**kwargs):

    a=kwargs.get("a",0)
    b=kwargs.get("b",1)

    print(a,b,c)

test(3,b=6)