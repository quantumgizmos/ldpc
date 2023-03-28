from ldpc2.codes import rep_code, ring_code
from ldpc2.noise_models import generate_bsc_error

H = ring_code(2)

print(H.toarray())

# error = generate_bsc_error(1, 0.2)
# print(error)

# print(H.toarray())