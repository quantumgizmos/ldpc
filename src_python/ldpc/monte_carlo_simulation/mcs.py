import numpy as np
import scipy.sparse as sp
from typing import Dict, Union
import datetime
import time
from tqdm import tqdm
from ldpc.noise_models import generate_bsc_error


class MonteCarloBscSimulation:
    """
    Class for running Monte Carlo simulations of a given error-correcting code using syndrome-based decoding.

    Parameters:
    -------------
    parity_check_matrix : Union[np.ndarray, sp.csr_matrix]
        The parity check matrix of the code
    error_rate : float
        The probability of bit-flip error for the channel
    Decoder : decoder object
        The decoder object used to decode the code
    target_run_count : int
        The number of Monte Carlo runs to be performed
    tqdm_disable : bool, optional
        Flag to disable the progress bar, by default False
    save_interval : int, optional
        Time interval (in seconds) for saving the intermediate results, by default 60
    seed : int, optional
        The seed for the random number generator, by default None. If set to None, the default
        numpy seed is used.

    Note:
    -----
    The Monte Carlo simulations are done using syndrome-based decoding relative to the zero codeword.

    """

    def __init__(
        self,
        parity_check_matrix: Union[np.ndarray, sp.csr_matrix] = None,
        error_rate: float = None,
        Decoder=None,
        target_run_count=1000,
        tqdm_disable=False,
        save_interval=60,
        seed=None,
        run=False,
    ) -> None:
        """
        Initializes a MonteCarloBscSimulation object with the given parameters.

        """
        if parity_check_matrix is None or not isinstance(
            parity_check_matrix, (np.ndarray, sp.csr_matrix)
        ):
            raise ValueError(
                f"parity_check_matrix should be of type np.ndarray or scipy.sparse.csr_matrix. Not {type(parity_check_matrix)}"
            )
        self.parity_check_matrix = parity_check_matrix

        if (
            error_rate is None
            or not isinstance(error_rate, float)
            or error_rate < 0
            or error_rate > 1
        ):
            raise ValueError(
                "Invalid error rate provided. The error rate should be a float with value between 0 and 1."
            )
        self.error_rate = error_rate

        if Decoder is None:
            raise ValueError("Invalid Decoder object provided.")
        self.Decoder = Decoder

        if not isinstance(target_run_count, int) or target_run_count <= 0:
            raise ValueError("Invalid target run count provided.")
        self.target_run_count = target_run_count

        if not isinstance(tqdm_disable, bool):
            raise ValueError("Invalid value for tqdm_disable flag.")
        self.tqdm_disable = tqdm_disable

        if not isinstance(save_interval, int) or save_interval <= 0:
            raise ValueError("Invalid save interval provided.")
        self.save_interval = save_interval

        if seed is None:
            self.seed = None
        else:
            if not isinstance(seed, int):
                raise ValueError(
                    "Invalid seed provided. Please provide a postive integer"
                )

            self.seed = seed
            np.random.seed(self.seed)

        self.run_count = 0
        self.fail_count = 0
        self.logical_error_rate = 0.0

        if run:
            self.run()

    def run(self) -> Dict:
        """
        Runs Monte Carlo simulations of the code.

        """
        self.start_date = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%A, %B %d, %Y %H:%M:%S"
        )

        # Set up the progress bar
        pbar = tqdm(
            range(self.run_count + 1, self.target_run_count + 1),
            disable=self.tqdm_disable,
            ncols=0,
        )

        self.fail_count = 0

        for self.run_count in pbar:
            # Generate a random bit-flip error
            error = generate_bsc_error(
                self.parity_check_matrix.shape[1], self.error_rate
            )

            # Calculate the corresponding syndrome and decoded codeword
            syndrome = self.parity_check_matrix @ error % 2  # calculate the syndrome

            decoding = self.Decoder.decode(
                syndrome
            )  # decode relative to the zero codeword

            # Check for decoding failure
            if not np.array_equal(decoding, error):
                self.fail_count += 1

            # Update the logical error rate and progress bar
            self.logical_error_rate = self.fail_count / self.run_count
            self.logical_error_rate_eb = np.sqrt(
                self.logical_error_rate * (1 - self.logical_error_rate) / self.run_count
            )
            pbar.set_description(
                f"Physical error rate: {100*self.error_rate:.2f}%; Logical error rate: {100*self.logical_error_rate:.2f}+-{100*self.logical_error_rate_eb:.2f}%"
            )
        return self.save()

    def save(self):
        output_dict = {}
        output_dict["logical_error_rate"] = self.logical_error_rate
        output_dict["logical_error_rate_eb"] = self.logical_error_rate_eb
        output_dict["error_rate"] = self.error_rate
        output_dict["run_count"] = self.run_count
        output_dict["fail_count"] = self.fail_count

        return output_dict
