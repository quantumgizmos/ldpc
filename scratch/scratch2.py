from ldpc2.codes import rep_code
from ldpc2.noise_models import generate_bsc_error

H = rep_code(100)

error = generate_bsc_error(100, 0.2)
print(error)

# print(H.toarray())