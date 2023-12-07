import numpy as np
from tqdm import tqdm


def random_binary_matrix(
    height: int = 10, width: int = 10, sparsity: float = 0.4, random_size: bool = False
):
    if random_size:
        height = np.random.randint(1, height)
        width = np.random.randint(1, width)

    mat = np.zeros(height * width).astype(np.uint8)

    for i, _ in enumerate(mat):
        if np.random.random() < sparsity:
            mat[i] = 1

    return mat.reshape((height, width))


def to_csr(mat: np.ndarray):
    m, n = mat.shape

    csr_string = "["
    for i in range(m):
        csr_string += "["
        first = True
        for j in range(n):
            if mat[i, j] == 1:
                if first != True:
                    csr_string += ","
                first = False
                csr_string += f"{j}"

        # if(!first): csr_string+="\b"
        csr_string += "]"
        if i != (m - 1):
            csr_string += ","
    csr_string += "]"

    return csr_string


def vector_to_string(vec):
    out = ""
    for i in range(len(vec)):
        out += f"{vec[i]}"
    return out


def add_rows_tests():
    output_file = open("gf2_add_test.csv", "w+")

    for j in range(100):
        for i in np.arange(0, 10, 0.5):
            height = np.random.randint(1, 40)
            width = np.random.randint(1, 40)
            pcm = random_binary_matrix(height=height, width=width, sparsity=0.1 * i)

            orig_pcm = to_csr(pcm)

            target_row = np.random.randint(height)
            add_row = np.random.randint(height)

            pcm[target_row] = (pcm[target_row] + pcm[add_row]) % 2

            final_pcm = to_csr(pcm)

            csv_string = (
                f"{height};{width};{orig_pcm};{target_row};{add_row};{final_pcm}"
            )
            print(csv_string, file=output_file)


def mulvec_tests():
    output_file = open("gf2_mulvec_test.csv", "w+")

    for j in tqdm(range(1000)):
        for i in np.arange(0, 10, 0.5):
            height = np.random.randint(1, 40)
            width = np.random.randint(1, 40)
            pcm = random_binary_matrix(
                height=height, width=width, sparsity=0.1 * np.random.randint(10)
            )

            vector = np.zeros(width).astype(int)
            for k in range(width):
                if np.random.random() < 0.1 * (10 - np.random.randint(10)):
                    vector[k] = 1

            output_vector = pcm @ vector % 2

            pcm = to_csr(pcm)
            vector = vector_to_string(vector)
            output_vector = vector_to_string(output_vector)

            # final_pcm = to_csr(pcm)

            csv_string = f"{height};{width};{pcm};{vector};{output_vector}"
            print(csv_string, file=output_file)


def mulvec_timing():
    output_file = open("gf2_mulvec_timing.csv", "w+")

    height = np.random.randint(1, 500)
    width = np.random.randint(1, 500)
    pcm = random_binary_matrix(
        height=height, width=width, sparsity=0.1 * np.random.randint(10)
    )
    pcm_csr = to_csr(pcm)

    csv_string = f"{height};{width};{pcm_csr}"
    print(csv_string, file=output_file)

    for j in tqdm(range(100)):
        for i in np.arange(0, 10, 0.5):
            vector = np.zeros(width).astype(int)
            for k in range(width):
                if np.random.random() < 0.1 * (10 - np.random.randint(10)):
                    vector[k] = 1

            output_vector = pcm @ vector % 2

            vector = vector_to_string(vector)
            output_vector = vector_to_string(output_vector)

            # final_pcm = to_csr(pcm)

            csv_string = f"{vector};{output_vector}"
            print(csv_string, file=output_file)


def matmul_tests():
    output_file = open("gf2_matmul_test.csv", "w+")

    for j in range(10):
        for i in np.arange(0, 10, 0.5):
            height = np.random.randint(1, 40)
            width = np.random.randint(1, 40)
            width2 = np.random.randint(1, 40)

            pcm1 = random_binary_matrix(
                height=height, width=width, sparsity=0.1 * np.random.randint(10)
            )
            height = width
            pcm2 = random_binary_matrix(
                height=height, width=width2, sparsity=0.1 * np.random.randint(10)
            )

            pcm3 = pcm1 @ pcm2 % 2

            pcms = [pcm1, pcm2, pcm3]

            csv_string = ""
            for pcm in pcms:
                m, n = pcm.shape
                csr = to_csr(pcm)

                csv_string += f"{m};{n};{csr};"

            print(csv_string, file=output_file)


def invert_tests():
    output_file = open("gf2_invert_test.csv", "w+")

    for j in range(100):
        n = np.random.randint(1, 40)

        pcm = np.identity(n)

        for _ in range(100):
            row1 = np.random.randint(n)
            row2 = np.random.randint(n)
            if not np.array_equal(pcm[row1], pcm[row2]):
                pcm[row1] = (pcm[row1] + pcm[row2]) % 2

        csv_string = ""
        m, n = pcm.shape
        csr = to_csr(pcm)
        csv_string += f"{m};{n};{csr};"
        print(csv_string, file=output_file)


def lu_solve_tests():
    output_file = open("gf2_lu_solve_test.csv", "a")

    for j in tqdm(range(100)):
        n = np.random.randint(1, 500)
        m = np.random.randint(1, 500)

        pcm = random_binary_matrix(
            height=m, width=n, sparsity=0.01
        )

        # for _ in range(np.random.randint(10)):
        #     pcm[np.random.randint(0, m)] = pcm[np.random.randint(0, m)]
        #     pcm[:, np.random.randint(0, n)] = np.zeros(m).astype(int)

        vector = np.zeros(n).astype(int)
        for k in range(n):
            if np.random.random() < 0.8 * (10 - np.random.randint(10)):
                vector[k] = 1

        output_vector = pcm @ vector % 2

        vector = vector_to_string(vector)
        output_vector = vector_to_string(output_vector)

        csv_string = ""
        m, n = pcm.shape
        csr = to_csr(pcm)
        csv_string += f"{m};{n};{csr};{output_vector}"
        print(csv_string, file=output_file)


if __name__ == "__main__":
    lu_solve_tests()
    # matmul_tests()
