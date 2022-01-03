#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mod2sparse.h"
#include "mod2sparse_extra.h"


int mod2sparse_rank(mod2sparse *A){
    int M,N;

    M=mod2sparse_rows(A);
    N=mod2sparse_cols(A);

    int nnf,rank;
    mod2sparse *L;
    mod2sparse *U;
    int *rows;
    int *cols;

    cols=chk_alloc(N,sizeof(*rows));
    rows=chk_alloc(M,sizeof(*cols));

    int abandon_number=0;  	/* Number of columns to abandon at some point *//* When to abandon these columns */
    int abandon_when=0;

    int submatrix_size;

    if(M==N){
        submatrix_size=M;
        // abandon_when=0;
    }

    if(M>N){
        submatrix_size=N;
        // abandon_when=N;
    }

    if(N>M){
        submatrix_size=M;
        // abandon_when=M;
    }
    
    L=mod2sparse_allocate(M,submatrix_size);
    U=mod2sparse_allocate(submatrix_size,N);

    mod2sparse_strategy strategy =Mod2sparse_first;/* Strategy to follow in picking rows/columns */

    nnf=mod2sparse_decomp
            (A,	/* Input matrix, M by N */
             submatrix_size,		/* Size of sub-matrix to find LU decomposition of */
             L,	/* Matrix in which L is stored, M by R */
             U,	/* Matrix in which U is stored, R by N */
             rows,		/* Array where row indexes are stored, M long */
             cols,		/* Array where column indexes are stored, N long */
            strategy,
            abandon_number,
            abandon_when
            );


    free(rows);
    free(cols);
    mod2sparse_free(L);
    mod2sparse_free(U);

    rank=submatrix_size-nnf;

    return rank;


}

void LU_forward_backward_solve
        (mod2sparse *L,
         mod2sparse *U,
         int *rows,
         int *cols,
         char *z,
         char *x)
{
    int N,R;
    char *forward_b;
    N=mod2sparse_cols(U);
    R=mod2sparse_cols(L);
    forward_b=chk_alloc(R,sizeof(*forward_b));

    for(int bit_no=0;bit_no<N;bit_no++) x[bit_no]=0;

    mod2sparse_forward_sub
            ( L,	/* Matrix that is lower triangular after reordering */
              rows,		/* Array of indexes (from 0) of rows for new order */
              z,		/* Vector on right of equation, also reordered */
              forward_b		/* Place to store solution */
            );
    mod2sparse_backward_sub
            ( U,	/* Matrix that is lower triangular after reordering */
              cols,		/* Array of indexes (from 0) of cols for new order */
              forward_b,		/* Vector on right of equation, also reordered */
              x		/* Place to store solution */
            );
    free(forward_b);
}





/* FIND AN LU DECOMPOSITION OF A SPARSE MATRIX. */

int mod2sparse_decomp_osd
        ( mod2sparse *A,	/* Input matrix, M by N */
          int R,		/* Size of sub-matrix to find LU decomposition of */
          mod2sparse *L,	/* Matrix in which L is stored, M by R */
          mod2sparse *U,	/* Matrix in which U is stored, R by N */
          int *rows,		/* Array where row indexes are stored, M long */
          int *cols		/* Array where column indexes are stored, N long */
        )
{

    int abandon_number=0;  	/* Number of columns to abandon at some point *//* When to abandon these columns */
    int abandon_when=0;
    mod2sparse_strategy strategy =Mod2sparse_first;/* Strategy to follow in picking rows/columns */

    int *rinv, *cinv, *acnt, *rcnt;
    mod2sparse *B;
    int M, N;

    mod2entry *e, *f, *fn, *e2;
    int i, j, k, cc, cc2, cc3, cr2, pr;
    int found, nnf;

    M = mod2sparse_rows(A);
    N = mod2sparse_cols(A);

    if (mod2sparse_cols(L)!=R || mod2sparse_rows(L)!=M
        || mod2sparse_cols(U)!=N || mod2sparse_rows(U)!=R)
    { fprintf (stderr,
               "mod2sparse_decomp: Matrices have incompatible dimensions\n");
        exit(1);
    }

    if (abandon_number>N-R)
    { fprintf(stderr,"Trying to abandon more columns than allowed\n");
        exit(1);
    }

    rinv = chk_alloc (M, sizeof *rinv);
    cinv = chk_alloc (N, sizeof *cinv);

    if (abandon_number>0)
    { acnt = chk_alloc (M+1, sizeof *acnt);
    }

    if (strategy==Mod2sparse_minprod)
    { rcnt = chk_alloc (M, sizeof *rcnt);
    }


    //these are the problematic functions!
    mod2sparse_clear(L);
    mod2sparse_clear(U);

    /* Copy A to B.  B will be modified, then discarded. */

    B = mod2sparse_allocate(M,N);
    mod2sparse_copy(A,B);

    /* Count 1s in rows of B, if using minprod strategy. */

    if (strategy==Mod2sparse_minprod)
    { for (i = 0; i<M; i++)
        { rcnt[i] = mod2sparse_count_row(B,i);
        }
    }

    /* Set up initial row and column choices. */

    for (i = 0; i<M; i++) rows[i] = rinv[i] = i;
    for (j = 0; j<N; j++) {

        cinv[cols[j]]=j;

    }
    /* Find L and U one column at a time. */

    nnf = 0;

    for (i = 0; i<R; i++)
    {
        /* Choose the next row and column of B. */

        switch (strategy)
        {
            case Mod2sparse_first:
            {
                found = 0;

                for (k = i; k<N; k++)
                { e = mod2sparse_first_in_col(B,cols[k]);
                    while (!mod2sparse_at_end(e))
                    { if (rinv[mod2sparse_row(e)]>=i)
                        { found = 1;
                            goto out_first;
                        }
                        e = mod2sparse_next_in_col(e);
                    }
                }

                out_first:
                break;
            }

            case Mod2sparse_mincol:
            {
                found = 0;

                for (j = i; j<N; j++)
                { cc2 = mod2sparse_count_col(B,cols[j]);
                    if (!found || cc2<cc)
                    { e2 = mod2sparse_first_in_col(B,cols[j]);
                        while (!mod2sparse_at_end(e2))
                        { if (rinv[mod2sparse_row(e2)]>=i)
                            { found = 1;
                                cc = cc2;
                                e = e2;
                                k = j;
                                break;
                            }
                            e2 = mod2sparse_next_in_col(e2);
                        }
                    }
                }

                break;
            }

            case Mod2sparse_minprod:
            {
                found = 0;

                for (j = i; j<N; j++)
                { cc2 = mod2sparse_count_col(B,cols[j]);
                    e2 = mod2sparse_first_in_col(B,cols[j]);
                    while (!mod2sparse_at_end(e2))
                    { if (rinv[mod2sparse_row(e2)]>=i)
                        { cr2 = rcnt[mod2sparse_row(e2)];
                            if (!found || cc2==1 || (cc2-1)*(cr2-1)<pr)
                            { found = 1;
                                pr = cc2==1 ? 0 : (cc2-1)*(cr2-1);
                                e = e2;
                                k = j;
                            }
                        }
                        e2 = mod2sparse_next_in_col(e2);
                    }
                }

                break;
            }

            default:
            { fprintf(stderr,"mod2sparse_decomp: Unknown stategy\n");
                exit(1);
            }
        }

        if (!found)
        { nnf += 1;
        }



        /* Update 'rows' and 'cols'.  Looks at 'k' and 'e' found above. */

        if (found)
        {
            // if (cinv[mod2sparse_col(e)]!=k) abort();
            if (cinv[mod2sparse_col(e)]!=k){
                printf("\n e: %i, k: %i",mod2sparse_col(e),k);
                printf("\nError. Exiting.");
                exit(1);
            }


            cols[k] = cols[i];
            cols[i] = mod2sparse_col(e);

            cinv[cols[k]] = k;
            cinv[cols[i]] = i;

            k = rinv[mod2sparse_row(e)];

            if (k<i) abort();

            rows[k] = rows[i];
            rows[i] = mod2sparse_row(e);

            rinv[rows[k]] = k;
            rinv[rows[i]] = i;
        }



        /* Update L, U, and B. */

        f = mod2sparse_first_in_col(B,cols[i]);

        while (!mod2sparse_at_end(f))
        {
            fn = mod2sparse_next_in_col(f);
            k = mod2sparse_row(f);
            if (rinv[k]>i)
            { mod2sparse_add_row(B,k,B,mod2sparse_row(e));
                if (strategy==Mod2sparse_minprod)
                { rcnt[k] = mod2sparse_count_row(B,k);
                }
                mod2sparse_insert(L,k,i);
            }
            else if (rinv[k]<i)
            { mod2sparse_insert(U,rinv[k],cols[i]);
            }
            else
            { mod2sparse_insert(L,k,i);
                mod2sparse_insert(U,i,cols[i]);
            }

            f = fn;
        }


        /* Get rid of all entries in the current column of B, just to save space. */

        for (;;)
        { f = mod2sparse_first_in_col(B,cols[i]);
            if (mod2sparse_at_end(f)) break;
            mod2sparse_delete(B,f);
        }

        /* Abandon columns of B with lots of entries if it's time for that. */

        if (abandon_number>0 && i==abandon_when)
        {
            for (k = 0; k<M+1; k++)
            { acnt[k] = 0;
            }
            for (j = 0; j<N; j++)
            { k = mod2sparse_count_col(B,j);
                acnt[k] += 1;
            }

            cc = abandon_number;
            k = M;
            while (acnt[k]<cc)
            { cc -= acnt[k];
                k -= 1;
                if (k<0) abort();
            }

            cc2 = 0;
            for (j = 0; j<N; j++)
            { cc3 = mod2sparse_count_col(B,j);
                if (cc3>k || cc3==k && cc>0)
                { if (cc3==k) cc -= 1;
                    for (;;)
                    { f = mod2sparse_first_in_col(B,j);
                        if (mod2sparse_at_end(f)) break;
                        mod2sparse_delete(B,f);
                    }
                    cc2 += 1;
                }
            }

            if (cc2!=abandon_number) abort();

            if (strategy==Mod2sparse_minprod)
            { for (j = 0; j<M; j++)
                { rcnt[j] = mod2sparse_count_row(B,j);
                }
            }
        }
    }

    /* Get rid of all entries in the rows of L past row R, after reordering. */

    for (i = R; i<M; i++)
    { for (;;)
        { f = mod2sparse_first_in_row(L,rows[i]);
            if (mod2sparse_at_end(f)) break;
            mod2sparse_delete(L,f);
        }
    }

    mod2sparse_free(B);
    free(rinv);
    free(cinv);
    if (strategy==Mod2sparse_minprod) free(rcnt);
    if (abandon_number>0) free(acnt);

    return nnf;
}


void mod2sparse_merge_vec(mod2sparse* m1, char *vec, int n, mod2sparse* m2) {

    if (mod2sparse_cols(m2) < mod2sparse_cols(m1) + 1 || mod2sparse_rows(m2) < mod2sparse_rows(m1)  || mod2sparse_rows(m2) < n) {
        printf("mod2sparse_merge_vec: the receiving matrix doesn't have the good dimensions");
    }

    mod2sparse_copy(m1,m2);

    int ncols = mod2sparse_cols(m1);
 
    for (int i = 0; i < n; i++){
        if (vec[i] != 0){ 
            mod2sparse_insert(m2,i,ncols);
        }
    }

}