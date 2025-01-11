# import scipy.sparse as sp
# import numpy as np
# from typing import List

# def generate_depolarising_noise(n:int, p: float = None, px: float = None,py: float = None,pz: float = None) -> np.ndarray:

#     if not isinstance(n, int) or n < 1:
#         raise ValueError("n should be a positive integer.")

#     if p is None:
#         if px is None or py is None or pz is None:
#             raise ValueError("Please provide either a single error rate `p` or three error rates `px`, `py` and `pz`.")

#     else:
#         if px is not None or py is not None or pz is not None:
#             raise ValueError("Please provide either a single error rate `p` or three error rates `px`, `py` and `pz`.")
#         else:
#             px = py = pz = p/3

#     assert px+py+pz <= 1, "The sum of the error rates should be less than or equal to 1."


#     return np.random.multinomial(1, [1-px-py-pz, px, py, pz], size=1)[0].astype(np.uint8)
