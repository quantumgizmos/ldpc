# Software for LDPC codes

The `mod2parse.h` and `mod2sparse.c` are adapted from Radford Neal's [Software for Low Density Parity Check Codes](https://www.cs.toronto.edu/~radford/ftp/LDPC-2012-02-11/index.html) package which was published under an MIT open source license (see COPYRIGHT file).

The following changes have been made to the origin files:

## Changes to `mod2sparse.h`
- Header guards have been added.
- The `mod2sparse.lr` attribute has been renamed to `mod2sparse.check_to_bit`.
- The `mod2sparse.pr` attribute has been renamed to `mod2sparse.bit_to_check`.
- Various functions relating to input/output in the origin file have been deleted.

## Changes to `mod2sparse.c`
- The `chk_alloc` function has been moved to `mod2sparse.c` from `alloc.c`.
- Various functions relating to input/output in the origin file have been deleted.