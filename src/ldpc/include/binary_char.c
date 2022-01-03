#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mod2sparse.h"
#include "binary_char.h"


void print_char_nonzero(char *val,int len)
  {
    printf("[ ");
    for(int bit_no=0;bit_no<len;bit_no++)
      {
        // printf("%i",val[bit_no]);
        if(val[bit_no]) printf("%i ",bit_no); 
      }
    printf(" ]");
  }

int bin_char_weight(char *val,int len)
  {
    int sum=0;
    for(int bit_no=0;bit_no<len;bit_no++)
      {
        if(val[bit_no]) sum++; 
      }
    return sum;
  }

int bin_char_equal(char *vec1, char *vec2, int len)
    {
        int sum=0;
        for(int i=0; i<len; i++)
            {
                sum+=vec1[i]^vec2[i];
                // printf("\n%i\n",sum);
            }
        if(sum==0) return 1;
        else return 0;
    }

int bin_char_is_zero(char *vec1, int len)
    {
        int sum=0;
        for(int i=0; i<len; i++)
            {
                sum+=vec1[i];
            }
        if(sum==0) return 1;
        else return 0;
    }

void print_char(char *val, int len)
  {
    for(int i=0;i<len;i++) printf("%i",val[i]);
  }


int bin_char_add(char *vec1, char *vec2, char *out_vec, int len)
    {
        for(int i=0; i<len; i++)
            {
                out_vec[i]=vec1[i]^vec2[i];
            }
    }

char *decimal_to_binary_reverse(int n,int k)
{
   char *binary_number;
   int divisor;
   int remainder;
   divisor=n;

   binary_number=chk_alloc(k,sizeof(char *));

   for(int i=0; i<k;i++)
   {
    remainder=divisor%2;
    binary_number[i]=remainder;
    divisor=divisor/2;
    if(divisor==0) break;
   }


   
   return  binary_number;
}

int hamming_difference(char *v1,char *v2,int len){
    int hamming_weight=0;

    for(int i=0;i<len;i++){
        hamming_weight+= v1[i]^v2[i];
    }

    return hamming_weight;

}

