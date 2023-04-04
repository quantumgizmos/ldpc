#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <memory>
#include <iterator>
#include <cmath> 


namespace util{

vector<uint8_t> decimal_to_binary(int decimal_nubmer,int binary_string_length, bool reverse=false)
{
   vector<uint8_t> binary_number;
   int divisor;
   int remainder;
   divisor=decimal_nubmer;

   binary_number.resize(binary_string_length);

   for(int i=0; i<binary_string_length;i++)
   {
        remainder=divisor%2;
        if(reverse) binary_number[i]=remainder;
        else binary_number[binary_string_length-i-1]=remainder;
        divisor=divisor/2;
        if(divisor==0) break;
   }

   return  binary_number;
}

vector<uint8_t> decimal_to_binary_reverse(int decimal_nubmer,int binary_string_length)
{
   return decimal_to_binary(decimal_nubmer,binary_string_length,true);
}

}//end namespace util


#endif