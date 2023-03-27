import pytest
import numpy as np
from scipy.sparse import csr_matrix
from ldpc2.bp_decoder import bp_decoder
from ldpc.codes import rep_code

def test_rep_code_ps():

    H = rep_code(3)

    bpd = bp_decoder(H,error_rate=0.1)
    assert bpd is not None
    print(bpd.bp_method)
    assert bpd.bp_method == "product_sum"
    assert bpd.schedule == "parallel"
    assert np.array_equal(bpd.error_channel,np.array([0.1, 0.1, 0.1]))


    bpd.decode(np.array([1, 1]))
    assert(np.array_equal(bpd.decoding, np.array([0, 1,0])))

    bpd.error_channel = np.array([0.1, 0, 0.1])
    assert np.array_equal(bpd.error_channel,np.array([0.1, 0, 0.1]))

    bpd.decode(np.array([1, 1]))
    assert(np.array_equal(bpd.decoding, np.array([1, 0, 1])))

def test_rep_code_ms():

    H = rep_code(3)

    bpd = bp_decoder(H,error_rate=0.1, bp_method='min_sum', ms_scaling_factor=1.0)
    assert bpd is not None
    assert bpd.bp_method == "minimum_sum"
    assert bpd.schedule == "parallel"
    assert np.array_equal(bpd.error_channel,np.array([0.1, 0.1, 0.1]))


    bpd.decode(np.array([1, 1]))
    assert(np.array_equal(bpd.decoding, np.array([0, 1,0])))

    bpd.error_channel = np.array([0.1, 0, 0.1])
    assert np.array_equal(bpd.error_channel,np.array([0.1, 0, 0.1]))

    bpd.decode(np.array([1, 1]))
    assert(np.array_equal(bpd.decoding, np.array([1, 0, 1])))
    

def test_rep_code_ps_serial():

    H = rep_code(3)

    bpd = bp_decoder(H,error_rate=0.1,schedule = "serial")
    assert bpd is not None
    assert bpd.bp_method == "product_sum"
    assert bpd.schedule == "serial"
    assert np.array_equal(bpd.error_channel,np.array([0.1, 0.1, 0.1]))
    assert np.array_equal(bpd.serial_schedule_order,np.array([0,1,2]))

    bpd.serial_schedule_order = np.array([2,1,0])

    assert np.array_equal(bpd.serial_schedule_order,np.array([2,1,0]))

    bpd.decode(np.array([1, 1]))
    assert(np.array_equal(bpd.decoding, np.array([0, 1,0])))

    bpd.error_channel = np.array([0.1, 0, 0.1])
    assert np.array_equal(bpd.error_channel,np.array([0.1, 0, 0.1]))

    bpd.decode(np.array([1, 1]))
    assert(np.array_equal(bpd.decoding, np.array([1, 0, 1])))

# if __name__ == '__main__':
#     # pass
#     test_rep_code_ps()
#     test_rep_code_ms()