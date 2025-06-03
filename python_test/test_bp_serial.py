import pytest
import numpy as np
from ldpc.codes import rep_code
from ldpc.bp_decoder import BpDecoder


def test_schedule_remains_same_with_manual_order():
    n = 5
    H = rep_code(n)
    manual_schedule_order = [4, 3, 2, 1, 0]

    decoder = BpDecoder(
        H,
        error_rate=0.1,
        max_iter=5,
        bp_method="minimum_sum",
        schedule="serial",
        serial_schedule_order=manual_schedule_order,
        # random_serial_schedule=False,
    )

    assert decoder.random_serial_schedule is False, "Random serial schedule should be disabled."
    assert decoder.serial_schedule_order is not None, "Serial schedule order should be specified."
    assert decoder.schedule == "serial", "Schedule should be set to 'serial'."


    syndrome = np.zeros(H.shape[0], dtype=np.uint8)
    syndrome[0] = 1  # Example syndrome to trigger decoding

    # Capture the schedule order after the first decode call
    decoder.decode(syndrome)
    first_schedule = decoder.serial_schedule_order

    # Capture the schedule order after the second decode call
    decoder.decode(syndrome)
    second_schedule = decoder.serial_schedule_order

    # Verify that the schedule remains the same
    assert np.array_equal(first_schedule, second_schedule), "Schedule changed despite being manually specified."
    assert np.array_equal(first_schedule, manual_schedule_order), "Schedule does not match the manually specified order."


def test_schedule_changes_with_random_serial_schedule():
    n = 10
    H = rep_code(n)

    decoder = BpDecoder(
        H,
        error_rate=0.1,
        max_iter=5,
        bp_method="minimum_sum",
        schedule="serial",
        random_serial_schedule=True,
        random_schedule_seed=0,  # Fixed seed for reproducibility
    )

    assert decoder.schedule == "serial"
    assert decoder.random_serial_schedule is True
    assert decoder.random_schedule_seed == 0

    syndrome = np.zeros(H.shape[0], dtype=np.uint8)
    syndrome[0] = 1  # Example syndrome to trigger decoding

    # Capture the schedule order after the first decode call
    decoder.decode(syndrome)
    first_schedule = decoder.serial_schedule_order

    # Capture the schedule order after the second decode call
    decoder.decode(syndrome)
    second_schedule = decoder.serial_schedule_order

    print("First schedule:", first_schedule)
    print("Second schedule:", second_schedule)

    # Verify that the schedule changes
    assert not np.array_equal(first_schedule, second_schedule), "Schedule did not change between decode calls."


def test_default_schedule_is_standard_and_constant():
    n = 10
    H = rep_code(n)

    decoder = BpDecoder(
        H,
        error_rate=0.1,
        max_iter=5,
        bp_method="minimum_sum",
        schedule="serial",
    )

    syndrome = np.zeros(H.shape[0], dtype=np.uint8)
    syndrome[0] = 1  # Example syndrome to trigger decoding

    # Capture the schedule order after the first decode call
    decoder.decode(syndrome)
    first_schedule = decoder.serial_schedule_order

    # Capture the schedule order after the second decode call
    decoder.decode(syndrome)
    second_schedule = decoder.serial_schedule_order

    print("First schedule:", first_schedule)
    print("Second schedule:", second_schedule)

    # Verify that the default schedule is standard {0, 1, 2, ...}
    default_schedule = np.arange(n)
    assert np.array_equal(first_schedule, default_schedule), "Default schedule is not standard {0, 1, 2, ...}."

    # Verify that the schedule remains constant
    assert np.array_equal(first_schedule, second_schedule), "Schedule changed despite random_serial_schedule being false."


def test_random_serial_schedule_with_default_seed():
    n = 10
    H = rep_code(n)

    decoder = BpDecoder(
        H,
        error_rate=0.1,
        max_iter=5,
        bp_method="minimum_sum",
        schedule="serial",
        random_serial_schedule=True,
    )

    assert decoder.random_serial_schedule is True, "Random serial schedule should be enabled."
    assert decoder.random_schedule_seed == 0, "Random schedule seed should be set to 0 by default."
    assert decoder.serial_schedule_order is not None, "Serial schedule order should be initialized."

    syndrome = np.zeros(H.shape[0], dtype=np.uint8)
    syndrome[0] = 1  # Example syndrome to trigger decoding

    # Capture the schedule order after the first decode call
    decoder.decode(syndrome)
    first_schedule = decoder.serial_schedule_order

    # Capture the schedule order after the second decode call
    decoder.decode(syndrome)
    second_schedule = decoder.serial_schedule_order

    # Verify that the schedule changes between decode calls
    assert not np.array_equal(first_schedule, second_schedule), "Schedule did not change between decode calls."


# if __name__ == "__main__":
#     pytest.main()

