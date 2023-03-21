#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <memory>
#include <iterator>

using namespace std;

namespace sparse_matrix{

    /**
     * @brief Base class for defining the node types for Sparse Matrices
     * 
     * This class defines the basic properties of a node in a sparse matrix such as its row index, column index,
     * and pointers to its neighboring nodes in the same row and column. Each node class that derives from this
     * base class will inherit these properties and add any additional properties as required by the specific
     * sparse matrix implementation.
     * 
     * @tparam DERIVED The derived class from EntryBase.
     */
    template <class DERIVED>
    class EntryBase {
        public:
            int row_index=-100; /**< The row index of the matrix element */
            int col_index=-100; /**< The column index of the matrix element */
            DERIVED* left = static_cast<DERIVED*>(this); /**< Pointer to the previous element in the row */
            DERIVED* right = static_cast<DERIVED*>(this); /**< Pointer to the next element in the row */
            DERIVED* up = static_cast<DERIVED*>(this); /**< Pointer to the previous element in the column */
            DERIVED* down = static_cast<DERIVED*>(this); /**< Pointer to the next element in the column */

            /**
             * @brief Resets the values of the entry to their default values.
             * 
             * This function resets the values of the entry to their default values. This is useful for when an 
             * entry is removed from the matrix and needs to be returned to its default state for re-use.
             */
            void reset(){
                row_index=-100;
                col_index=-100;
                left = static_cast<DERIVED*>(this);
                right = static_cast<DERIVED*>(this);
                up = static_cast<DERIVED*>(this);
                down = static_cast<DERIVED*>(this);
            }

            /**
             * @brief Checks if the entry is at the end of the list
             * 
             * This function checks if the entry is at the end of the list by checking if its row index is equal
             * to -100. If it is, then the function returns true to indicate that the entry is at the end of the
             * list.
             * 
             * @return True if the entry is at the end, false otherwise
             */
            bool at_end(){
                if(row_index==-100) return true;
                else return false;
            }

            /**
             * @brief Returns a string representation of the entry
             * 
             * This function returns a string representation of the entry. In this implementation, the function
             * always returns "1", but in other implementations, this function could be used to return the value
             * stored in the entry or some other useful information.
             * 
             * @return The string representation of the entry
             */
            string str(){
                return "1";
            }

        ~EntryBase(){};
    };



    template <class T>
    class SparseMatrixEntry: public EntryBase<SparseMatrixEntry<T>> {
        public:
            T value = T(0); // the value structure we are storing at each matrix location. We can define this as any object, and overload operators.
        ~SparseMatrixEntry(){};
        
        string str(){
            return std::to_string(this->value);
        }

    };

    template <class ENTRY_OBJ>
    class SparseMatrixBase {
    public:
        int m,n; //m: check-count; n: bit-count
        int node_count;
        int entry_block_size;
        int released_entry_count;
        vector<ENTRY_OBJ*> entries;
        vector<ENTRY_OBJ*> removed_entries;       
        vector<ENTRY_OBJ*> row_heads; //starting point for each row
        vector<ENTRY_OBJ*> column_heads; //starting point for each column

        // vector<ENTRY_OBJ*> matrix_entries;
        
        SparseMatrixBase(int check_count, int bit_count){
            this->m=check_count;
            this->n=bit_count;
            this->entry_block_size = this->m+this->n;
            this->released_entry_count=0;
            allocate();
        }

        ~SparseMatrixBase(){
            for(auto e: this->entries) delete e;
            this->entries.clear();


        }

        /**
         * @brief Allocates a new entry object and returns a pointer to it.
         *
         * If there are any entries that have been removed from the matrix, the function returns the last 
         * removed entry. Otherwise, if there is space in the entries vector, a new entry is allocated at 
         * the end of the vector. If the entries vector is full, the function allocates a new block of
         * entries and returns the first entry in the new block.
         *
         * @return A pointer to a new entry object.
         */
        ENTRY_OBJ* allocate_new_entry(){
            if(this->removed_entries.size()!=0){
                auto e = this->removed_entries.back();
                this->removed_entries.pop_back();
                return e; 
            }
            if(this->released_entry_count==this->entries.size()){
                int new_size = this->entries.size()+this->entry_block_size;
                for(int i = 0; i<this->entry_block_size; i++){
                    this->entries.push_back(new ENTRY_OBJ());
                }
            }

            auto e = this->entries[this->released_entry_count];
            this->released_entry_count++;

            return e;
        }

        /**
         * @brief Returns the number of non-zero entries in the matrix.
         *
         * @return The number of non-zero entries in the matrix.
         */
        int entry_count(){
            return this->released_entry_count - this->n - this->m - this->removed_entries.size();
        }

        /**
         * @brief Computes the sparsity of the matrix
         *
         * The sparsity of a matrix is defined as the ratio of the number of zero elements in the
         * matrix to the total number of elements in the matrix. This function computes the
         * sparsity of the matrix represented by this object, and returns the result as a double value.
         *
         * @return The sparsity of the matrix as a double value.
         */
        double sparsity(){
            return this->entry_count()/(this->m*this->n);
        }

        /**
         * @brief Allocate space for an mxn matrix and initialize the row and column headers.
         * 
         * @details This function allocates space for an mxn matrix, and initializes the row and column headers 
         * to self-referential values with row and column indices of -100, respectively, to indicate that they 
         * are not value elements. This function must be called before any entries can be inserted into the matrix.
         * 
         */
        void allocate(){
                
            this->row_heads.resize(this->m);
            this->column_heads.resize(this->n);

            for(int i=0; i<this->m; i++){
                ENTRY_OBJ* row_entry;
                row_entry = this->allocate_new_entry();
                row_entry->row_index = -100; //we set the row head index to -100 to indicate is not a value element
                row_entry->col_index = -100;
                row_entry->right = row_entry; //at first the head elements point to themselves
                row_entry->left = row_entry;
                row_entry->up = row_entry;
                row_entry->down = row_entry;
                this->row_heads[i] = row_entry;

            }

            for(int i=0; i<this->n; i++){
                ENTRY_OBJ* column_entry; 
                column_entry = this->allocate_new_entry();
                column_entry->row_index = -100;
                column_entry->col_index = -100;
                column_entry->right = column_entry;
                column_entry->left = column_entry;
                column_entry->up = column_entry;
                column_entry->down = column_entry;
                this->column_heads[i] = column_entry;
            }
        }   


        void swap_rows(int i, int j){
            auto tmp1 = this->row_heads[i];
            auto tmp2 = this->row_heads[j];
            this->row_heads[j] = tmp1;
            this->row_heads[i] = tmp2;
            for(auto e: iterate_row(i)) e->row_index=i;
            for(auto e: iterate_row(j)) e->row_index=j;
        }

        void reorder_rows(vector<int> rows){

            vector<ENTRY_OBJ*> temp_row_heads = this->row_heads;
            // for(int i = 0; i<m; i++) temp_row_heads.push_back(row_heads[i]);
            for(int i = 0; i<m; i++){
                this->row_heads[i] = temp_row_heads[rows[i]];
                for(auto e: this->iterate_row(i)){
                    e->row_index = i;
                }
            }

        }

        void swap_columns(int i, int j){
            auto tmp1 = this->column_heads[i];
            auto tmp2 = this->column_heads[j];
            this->column_heads[j] = tmp1;
            this->column_heads[i] = tmp2;
            for(auto e: this->iterate_column(i)) e->col_index=i;
            for(auto e: this->iterate_column(j)) e->col_index=j;
        }

        int get_row_degree(int row){
            return abs(this->row_heads[row]->col_index)-100;
        }

        int get_col_degree(int col){
            return abs(this->column_heads[col]->col_index)-100;
        }

        void remove_entry(int i, int j){
            auto e = this->get_entry(i,j);
            this->remove(e);
        }

        void remove(ENTRY_OBJ* e){
            if(!e->at_end()){
                auto e_left = e->left;
                auto e_right = e-> right;
                auto e_up = e->up;
                auto e_down = e ->down;
                e_left->right = e_right;
                e_right->left = e_left;
                e_up ->down = e_down;
                e_down -> up = e_up;


                /*updates the row/column weights. Note this information is stored in the
                `ENTRY_OBJ.col_index` field as a negative number (to avoid confusion with
                an actual col index. To get the row/column weights use the `get_row_weight()` and
                `get_col_weight()` functions.)*/
                this->row_heads[e->row_index]->col_index++;
                this->column_heads[e->col_index]->col_index++;

                //Reset the entry to default values before pushing to the buffer.
                e->reset();
                this->removed_entries.push_back(e);

            

            

            }
        }

        ENTRY_OBJ* insert_entry(int j, int i){
            if(j>=this->m || i>=this->n || j<0 || i<0) throw invalid_argument("Index i or j is out of bounds"); 
            
            auto left_entry = this->row_heads[j];
            auto right_entry = this->row_heads[j];
            
            for(auto entry: iterate_row(j)){
                
                if(entry->col_index == i){
                    return entry;
                }
                
                if(entry->col_index < i) left_entry = entry;
                
                if(entry->col_index > i) {
                    right_entry = entry;
                    break;
                }
            
            }

            auto up_entry = this->column_heads[i];
            auto down_entry = this->column_heads[i];

            for(auto entry: this->iterate_column(i)){
                if(entry->row_index < j) up_entry = entry;
                if(entry->row_index > j) {
                    down_entry = entry;
                    break;
                }
            }

            auto e = this->allocate_new_entry();
            node_count++;
            e->row_index = j;
            e->col_index = i;
            e->right = right_entry;
            e->left = left_entry;
            e->up = up_entry;
            e->down = down_entry;

            left_entry->right = e;
            right_entry->left = e;
            up_entry->down = e;
            down_entry->up = e;

            /*updates the row/column weights. Note this information is stored in the
            `ENTRY_OBJ.col_index` field as a negative number (to avoid confusion with
            an actual col index. To get the row/column weights use the `get_row_weight()` and
            `get_col_weight()` functions.)*/

            this->row_heads[e->row_index]->col_index--;
            this->column_heads[e->col_index]->col_index--;

            // matrix_entries.push_back(e);
            return e;     
        }

        ENTRY_OBJ* get_entry(int j, int i){
            if(j>=this->m || i>=this->n || j<0 || i<0) throw invalid_argument("Index i or j is out of bounds"); 
            for(auto e: this->iterate_column(i)){
                if(e->row_index==j) return e;
            }
            return this->column_heads[i];
            
        }

        ENTRY_OBJ* insert_row(int row_index, vector<int>& col_indices){
            for(auto j: col_indices){
                this->insert_entry(row_index,j);
            }
            return this->row_heads[row_index];
        }

        vector<vector<int>> nonzero_coordinates(){

            vector<vector<int>> nonzero;

            this->node_count = 0;

            for(int i = 0; i<this->m; i++){
                for(auto e: this->iterate_row(i)){
                    if(e->value == 1){
                        this->node_count += 1;
                        vector<int> coord;
                        coord.push_back(e->row_index);
                        coord.push_back(e->col_index);
                        nonzero.push_back(coord);
                    }
                }
            }

            return nonzero;

        }

        vector<vector<int>> nonzero_rows(){

            vector<vector<int>> nonzero;

            this->node_count = 0;

            for(int i = 0; i<this->m; i++){
                vector<int> row;
                for(auto e: this->iterate_row(i)){
                    this->node_count += 1;
                    row.push_back(e->col_index);
                }
                nonzero.push_back(row);
            }

            return nonzero;

        }



        template <class DERIVED>
        class Iterator{
            public:
                using iterator_category = std::forward_iterator_tag;
                using difference_type   = std::ptrdiff_t;
                SparseMatrixBase<ENTRY_OBJ>* matrix;
                ENTRY_OBJ* e;
                ENTRY_OBJ* head;
                // int index = -1;
                Iterator(SparseMatrixBase<ENTRY_OBJ>* mat){
                    matrix = mat;
                }
                ~Iterator(){};
                DERIVED end(){
                    return static_cast<DERIVED&>(*this);
                }
                ENTRY_OBJ* operator*() const { return e; };
                friend bool operator== (const Iterator& a, const Iterator& b) { return a.e == b.head; };
                friend bool operator!= (const Iterator& a, const Iterator& b) { return a.e != b.head; };
        };

        class RowIterator: public Iterator<RowIterator>{
            public:
                typedef Iterator<RowIterator> BASE;
                using BASE::e; using BASE::matrix; using BASE::head;
                RowIterator(SparseMatrixBase<ENTRY_OBJ>* mat) : BASE::Iterator(mat){}

                RowIterator begin(){
                    e=head->right;
                    return *this;
                }
                RowIterator& operator++(){
                    e=e->right;
                    return *this;
                }

                RowIterator operator[](int i)
                {
                    head = matrix->row_heads[i];
                    return *this;
                }
        };

        class ReverseRowIterator: public Iterator<ReverseRowIterator>{
            public:
                typedef Iterator<ReverseRowIterator> BASE;
                using BASE::e; using BASE::matrix;  using BASE::head;
                ReverseRowIterator(SparseMatrixBase<ENTRY_OBJ>* mat) : BASE::Iterator(mat){}

                ReverseRowIterator begin(){
                    e=head->left;
                    return *this;
                }
                ReverseRowIterator& operator++(){
                    e=e->left;
                    return *this;
                }

                ReverseRowIterator operator[](int i)
                {
                    head = matrix->row_heads[i];
                    return *this;
                }
        };

        class ColumnIterator: public Iterator<ColumnIterator>{
            public:
                typedef Iterator<ColumnIterator> BASE;
                using BASE::e; using BASE::matrix;  using BASE::head;
                ColumnIterator(SparseMatrixBase<ENTRY_OBJ>* mat) : BASE::Iterator(mat){}

                ColumnIterator begin(){
                    e=head->down;
                    return *this;
                }
                ColumnIterator& operator++(){
                    e=e->down;
                    return *this;
                }
                ColumnIterator operator[](int i)
                {
                    head = matrix->column_heads[i];
                    return *this;
                }
        };

        class ReverseColumnIterator: public Iterator<ReverseColumnIterator>{
            public:
                typedef Iterator<ReverseColumnIterator> BASE;
                using BASE::e; using BASE::matrix;  using BASE::head;
                ReverseColumnIterator(SparseMatrixBase<ENTRY_OBJ>* mat) : BASE::Iterator(mat){}

                ReverseColumnIterator begin(){
                    e=head->up;
                    return *this;
                }
                ReverseColumnIterator& operator++(){
                    e=e->up;
                    return *this;
                }
                ReverseColumnIterator operator[](int i)
                {
                    head = matrix->column_heads[i];
                    return *this;
                }
        };


        RowIterator iterate_row(int i){
            if(i<0 || i>=m) throw invalid_argument("Iterator index out of bounds");
            return RowIterator(this)[i];
        }

        ReverseRowIterator reverse_iterate_row(int i){
            if(i<0 || i>=m) throw invalid_argument("Iterator index out of bounds");
            return ReverseRowIterator(this)[i];
        }

        ColumnIterator iterate_column(int i){
            if(i<0 || i>=n) throw invalid_argument("Iterator index out of bounds");
            return ColumnIterator(this)[i];
        }

        ReverseColumnIterator reverse_iterate_column(int i){
            if(i<0 || i>=n) throw invalid_argument("Iterator index out of bounds");
            return ReverseColumnIterator(this)[i];
        }

    };


    template <class T, template<class> class ENTRY_OBJ=SparseMatrixEntry>
    class SparseMatrix: public SparseMatrixBase<ENTRY_OBJ<T>> {
    typedef SparseMatrixBase<ENTRY_OBJ<T>> BASE;
    public:
        SparseMatrix(int m, int n): BASE::SparseMatrixBase(m,n){};
        ENTRY_OBJ<T>* insert_entry(int i, int j, T val = T(1)){
            auto e = BASE::insert_entry(i,j);
            e->value = val;
            return e;
        }

        ENTRY_OBJ<T>* insert_row(int row_index, vector<int>& col_indices, vector<T>& values){
            BASE::insert_row(row_index,col_indices);
            int i = 0;
            for(auto e: this->iterate_row(row_index)){
                e->value = values[i];
                i++; 
            }
            return this->row_heads[row_index];
        }

        ~SparseMatrix(){};
    };

}


#endif