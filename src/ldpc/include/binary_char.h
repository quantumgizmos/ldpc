#ifndef BINARY_CHAR_H
#define BINARY_CHAR_H

void print_char_nonzero(char *val,int len);
int bin_char_equal(char *vec1, char *vec2, int len);
int bin_char_is_zero(char *vec1, int len);
void print_char(char *val, int len);
int bin_char_add(char *vec1, char *vec2, char *out_vec, int len);
char *decimal_to_binary_reverse(int n,int K);
int bin_char_weight(char *val,int len);
int hamming_difference(char *v1,char *v2,int len);

#endif 