import re


def remove_function_implementations(input_file_path, output_file_path):
    with open(input_file_path, "r") as input_file:
        input_lines = input_file.readlines()

    output_lines = []
    for line in input_lines:
        if re.match(r"^def .*\(.*\):", line):
            # function declaration - keep it
            output_lines.append(line)
        elif re.match(r"^cdef .*\(.*\):", line):
            # cdef function declaration - remove "cdef" and keep it
            output_lines.append(line.replace("cdef ", ""))
        elif re.match(r"^class .*:", line):
            # class declaration - keep it
            output_lines.append(line)
        elif re.match(r"^    def .*\(.*\):", line):
            # indented function declaration - keep it
            output_lines.append(line)
        elif re.match(r"^    cdef .*\(.*\):", line):
            # indented cdef function declaration - remove "cdef" and keep it
            output_lines.append(line.replace("cdef ", ""))
        elif re.match(r"^\s+\"\"\"", line):
            # docstring - keep it
            output_lines.append(line)
            # skip all subsequent lines until the closing quotes
            while not re.match(r"^\s+\"\"\"", input_lines[0]):
                input_lines.pop(0)
        else:
            # everything else - replace with "pass"
            output_lines.append("    pass\n")

    with open(output_file_path, "w") as output_file:
        output_file.writelines(output_lines)


if __name__ == "__main__":
    remove_function_implementations("_bp_decoder.pyx", "__init__.pyi")
