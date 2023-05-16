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


/**
 * @brief Template base class for implementing sparse matrices in a doubly linked list format
 * 
 * @tparam ENTRY_OBJ The entry object class that the sparse matrix will use
 * for its entries. This class should contain fields for row index, column index,
 * and value.
 * 
 * This class allows for the construction of sparse matrices with custom data types by
 * passing node objects via the ENTRY_OBJ template parameter. The matrix is stored as a
 * doubly linked list, where each row and column is represented by a linked list of entries.
 * Each entry contains a reference to the next and previous entries in its row and column,
 * respectively. This format allows for efficient insertion and deletion of entries in the matrix,
 * especially when the matrix is large and sparse.
 */
template <class ENTRY_OBJ>
class SparseMatrixBase {
public:
    int m,n; //m: check-count; n: bit-count
    int node_count;
    int entry_block_size;
    int allocated_entry_count;
    int released_entry_count;
    int block_position;
    int block_idx;
    vector<ENTRY_OBJ*> entries;
    vector<ENTRY_OBJ*> removed_entries;       
    vector<ENTRY_OBJ*> row_heads; //starting point for each row
    vector<ENTRY_OBJ*> column_heads; //starting point for each column

    // vector<ENTRY_OBJ*> matrix_entries;

    /**
     * @brief Constructs a sparse matrix with the given dimensions
     * 
     * @param check_count The number of rows in the matrix
     * @param bit_count The number of columns in the matrix
     */
    SparseMatrixBase(int check_count, int bit_count, int entry_count = 0){
        this->m=check_count;
        this->n=bit_count;
        this->block_idx=-1;
        this->released_entry_count=0;
        this->allocated_entry_count=0;
        this->entry_block_size = this->m+this->n + entry_count;
        allocate();
        this->entry_block_size = this->m+this->n;
    }

    /**
     * @brief Destructor for SparseMatrixBase. Frees memory occupied by entries.
     */
    ~SparseMatrixBase(){
        for(auto entry_block: this->entries) delete[] entry_block;
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
        // if there are any previously removed entries, use them instead of creating new ones
        if(this->removed_entries.size()!=0){
            auto e = this->removed_entries.back();
            this->removed_entries.pop_back();
            return e; 
        }
        // if there are no previously removed entries, create a new one
        // if there is no space for the new entry, add a new block of entries
        if(this->released_entry_count==this->allocated_entry_count){
            this->entries.push_back(new ENTRY_OBJ[this->entry_block_size]());
            this->allocated_entry_count+=this->entry_block_size;
            this->block_idx++;
            this->block_position=0;
        }



        // use the next available entry in the pool
        auto e = &this->entries[block_idx][this->block_position];
        this->block_position++;
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
    * @brief Allocates memory for the row and column header nodes.
    *
    * This function resizes the row_heads and column_heads vectors to m and n respectively.
    * For each row and column header node, it allocates memory for a new entry object, sets
    * the row and column indices to -100 to indicate it is not a value element, and sets the right,
    * left, up, and down pointers to point to itself since there are no other nodes in the same row or column yet.
    */
    void allocate(){
            
        this->row_heads.resize(this->m); // resize row_heads vector to m
        this->column_heads.resize(this->n); // resize column_heads vector to n

        // create and initialize each row header node
        for(int i=0; i<this->m; i++){
            ENTRY_OBJ* row_entry; // pointer to a new entry object
            row_entry = this->allocate_new_entry(); // allocate memory for a new entry object
            row_entry->row_index = -100; // set row index to -100 to indicate it is not a value element
            row_entry->col_index = -100; // set col index to -100 to indicate it is not a value element
            row_entry->right = row_entry; // point to itself since there are no other nodes in the same row yet
            row_entry->left = row_entry; // point to itself since there are no other nodes in the same row yet
            row_entry->up = row_entry; // point to itself since there are no other nodes in the same column yet
            row_entry->down = row_entry; // point to itself since there are no other nodes in the same column yet
            this->row_heads[i] = row_entry; // add the new row header node to the row_heads vector
        }

        // create and initialize each column header node
        for(int i=0; i<this->n; i++){
            ENTRY_OBJ* column_entry; // pointer to a new entry object
            column_entry = this->allocate_new_entry(); // allocate memory for a new entry object
            column_entry->row_index = -100; // set row index to -100 to indicate it is not a value element
            column_entry->col_index = -100; // set col index to -100 to indicate it is not a value element
            column_entry->right = column_entry; // point to itself since there are no other nodes in the same column yet
            column_entry->left = column_entry; // point to itself since there are no other nodes in the same column yet
            column_entry->up = column_entry; // point to itself since there are no other nodes in the same row yet
            column_entry->down = column_entry; // point to itself since there are no other nodes in the same row yet
            this->column_heads[i] = column_entry; // add the new column header node to the column_heads vector
        }
    }   



    /**
     * @brief Swaps two rows of the matrix
     * 
     * This function swaps rows i and j of the matrix. All row entries must be updated accordingly.
     * 
     * @param i The index of the first row to swap
     * @param j The index of the second row to swap
     */
    void swap_rows(int i, int j){
        auto tmp1 = this->row_heads[i]; // store the head element of row i in a temporary variable
        auto tmp2 = this->row_heads[j]; // store the head element of row j in a temporary variable
        this->row_heads[j] = tmp1; // set the head element of row j to the head element of row i
        this->row_heads[i] = tmp2; // set the head element of row i to the head element of row j
        for(auto e: iterate_row(i)) e->row_index=i; // update the row index of all elements in row i to j
        for(auto e: iterate_row(j)) e->row_index=j; // update the row index of all elements in row j to i
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

    /**
     * Swaps two columns in the sparse matrix.
     * 
     * @param i The index of the first column to swap.
     * @param j The index of the second column to swap.
     */
    void swap_columns(int i, int j){
        auto tmp1 = this->column_heads[i];
        auto tmp2 = this->column_heads[j];
        this->column_heads[j] = tmp1;
        this->column_heads[i] = tmp2;
        // update the column indices for all entries in columns i and j
        for(auto e: this->iterate_column(i)) e->col_index=i;
        for(auto e: this->iterate_column(j)) e->col_index=j;
    }

    /**
     * @brief Gets the number of non-zero entries in a row of the matrix.
     * 
     * This function returns the degree of a given row in the matrix, i.e., the number of non-zero entries in the row.
     * 
     * @param row The index of the row to get the degree of.
     * @return The number of non-zero entries in the row.
     */
    int get_row_degree(int row){
        return abs(this->row_heads[row]->col_index)-100;
    }

    /**
     * @brief Gets the number of non-zero entries in a column of the matrix.
     * 
     * This function returns the degree of a given column in the matrix, i.e., the number of non-zero entries in the column.
     * 
     * @param col The index of the column to get the degree of.
     * @return The number of non-zero entries in the column.
     */
    int get_col_degree(int col){
        return abs(this->column_heads[col]->col_index)-100;
    }

    /**
     * @brief Removes an entry from the matrix.
     * 
     * This function removes a given entry from the matrix. The entry is identified by its row and column indices.
     * 
     * @param i The row index of the entry to remove.
     * @param j The column index of the entry to remove.
     */
    void remove_entry(int i, int j){
        auto e = this->get_entry(i,j);
        this->remove(e);
    }

    /**
     * @brief Removes an entry from the matrix and updates the row/column weights
     * 
     * @param e Pointer to the entry object to be removed
     */
    void remove(ENTRY_OBJ* e){
        // Check if the entry is not already at the end of the row or column.
        if(!e->at_end()){
            // Store pointers to the adjacent entries.
            auto e_left = e->left;
            auto e_right = e->right;
            auto e_up = e->up;
            auto e_down = e ->down;

            // Update pointers of the adjacent entries to remove the entry from the linked list.
            e_left->right = e_right;
            e_right->left = e_left;
            e_up ->down = e_down;
            e_down -> up = e_up;

            /* Update the row/column weights. Note that this information is stored in the
            ENTRY_OBJ.col_index field as a negative number (to avoid confusion with
            an actual column index). To get the row/column weights, use the get_row_weight()
            and get_col_weight() functions. */
            this->row_heads[e->row_index]->col_index++;
            this->column_heads[e->col_index]->col_index++;

            // Reset the entry to default values before pushing it to the buffer.
            e->reset();
            // Store the removed entry in the buffer for later reuse.
            this->removed_entries.push_back(e);
        }
    }

    /**
    * @brief Inserts a new entry in the matrix at position (j, i).
    * 
    * @param j The row index of the new entry.
    * @param i The column index of the new entry.
    * @return A pointer to the newly created ENTRY_OBJ object.
    * @throws std::invalid_argument if either i or j is out of bounds.
    * 
    * This function inserts a new entry in the matrix at position (j, i). If an entry
    * already exists at that position, this function simply returns a pointer to it. 
    * Otherwise, it creates a new entry and inserts it into the matrix, linking it to 
    * the neighboring entries to maintain the doubly linked structure. This function 
    * also updates the row and column weights of the matrix. The row and column weights 
    * are stored as negative integers in the col_index field of the ENTRY_OBJ object.
    * To retrieve the actual weights, use the get_row_weight() and get_col_weight()
    * functions.
    */
    ENTRY_OBJ* insert_entry(int j, int i){
        // Check if indices are within bounds
        if(j>=this->m || i>=this->n || j<0 || i<0) throw invalid_argument("Index i or j is out of bounds"); 
                
        // Find the left and right entries in the jth row of the matrix
        auto left_entry = this->row_heads[j];
        auto right_entry = this->row_heads[j];
        for(auto entry: iterate_row(j)){
            if(entry->col_index == i){
                // Entry already exists at this position
                return entry;
            }
            if(entry->col_index < i) left_entry = entry;
            if(entry->col_index > i) {
                right_entry = entry;
                break;
            }
        }

        // Find the up and down entries in the ith column of the matrix
        auto up_entry = this->column_heads[i];
        auto down_entry = this->column_heads[i];
        for(auto entry: this->iterate_column(i)){
            if(entry->row_index < j) up_entry = entry;
            if(entry->row_index > j) {
                down_entry = entry;
                break;
            }
        }

        // Create and link the new entry
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

        // Update row and column weights
        this->row_heads[e->row_index]->col_index--;
        this->column_heads[e->col_index]->col_index--;

        // Return a pointer to the newly created entry
        return e;     
    }



    /**
     * Get an entry at row j and column i.
     *
     * @param j The row index
     * @param i The column index
     * @return a pointer to the entry, or a pointer to the corresponding column head.
     * @throws std::invalid_argument if j or i are out of bounds.
     */
    ENTRY_OBJ* get_entry(int j, int i){
        if(j>=this->m || i>=this->n || j<0 || i<0) throw invalid_argument("Index i or j is out of bounds");

        // Iterate over the column at index i and check each entry's row index.
        // If the row index matches j, return that entry.
        for(auto e: this->iterate_column(i)){
            if(e->row_index==j) return e;
        }

        // If no entry is found, return the column head at index i.
        return this->column_heads[i];
    }


    /**
     * Insert a new row at row_index with entries at column indices col_indices.
     *
     * @param row_index The index of the row to insert.
     * @param col_indices A vector of indices indicating which columns to insert entries into.
     * @return a pointer to the row head entry for the newly inserted row.
     */
    ENTRY_OBJ* insert_row(int row_index, vector<int>& col_indices){
        // Insert an entry at each specified column index.
        for(auto j: col_indices){
            this->insert_entry(row_index,j);
        }

        // Return the row head at row_index.
        return this->row_heads[row_index];
    }


    /**
     * @brief Returns the coordinates of all non-zero entries in the matrix.
     *
     * @return Vector of vectors, where each inner vector represents the row and column indices of a non-zero entry.
     */
    vector<vector<int>> nonzero_coordinates(){

        vector<vector<int>> nonzero;

        // Initialize node count to 0
        this->node_count = 0;

        // Iterate through all rows and columns to find non-zero entries
        for(int i = 0; i<this->m; i++){
            for(auto e: this->iterate_row(i)){
                if(e->value == 1){
                    // Increment node count and add non-zero entry coordinates to vector
                    this->node_count += 1;
                    vector<int> coord;
                    coord.push_back(e->row_index);
                    coord.push_back(e->col_index);
                    nonzero.push_back(coord);
                }
            }
        }

        // Return vector of non-zero entry coordinates
        return nonzero;

    }

    /**
    * Returns a vector of vectors, where each vector contains the column indices of the non-zero entries in a row.
    * @return A vector of vectors, where each vector contains the column indices of the non-zero entries in a row.
    */
    vector<vector<int>> nonzero_rows(){

        vector<vector<int>> nonzero;

        this->node_count = 0; //reset node_count to 0

        //iterate over the rows of the matrix
        for(int i = 0; i<this->m; i++){
            vector<int> row;
            
            //iterate over the entries in the current row
            for(auto e: this->iterate_row(i)){
                this->node_count += 1; //increment node_count
                row.push_back(e->col_index); //add the column index of the current entry to the current row vector
            }
            nonzero.push_back(row); //add the current row vector to the vector of non-zero rows
        }

        return nonzero;

    }




    /**
     * @brief Base class for implementing iterators for sparse matrices
     *
     * This class provides a basic implementation of the required methods for a forward
     * iterator. It is intended to be used as a base class for iterators that inherit from
     * it via the Curiously Recurring Template Pattern (CRTP). The derived classes must
     * implement a `next()` method, which updates the iterator to point to the next element
     * in the matrix.
     *
     * @tparam DERIVED The derived class that implements the `next()` method. This should
     * be a template parameter of the derived class, using the `Iterator` class as the
     * template argument.
     * @tparam ENTRY_OBJ The entry object class that the sparse matrix uses for its entries.
     * This class should contain fields for row index, column index, and value.
     */
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


    /**
     * @brief An iterator class that iterates over rows of a sparse matrix in a doubly linked list format.
     * 
     * This class inherits from the Iterator class and is designed to work with SparseMatrixBase and its
     * subclasses. It is used to iterate over the rows of a sparse matrix in a doubly linked list format.
     * 
     * @tparam ENTRY_OBJ The entry object class that the sparse matrix will use for its entries. This class
     * should contain fields for row index, column index, and value.
     */
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

    /**
     * @brief A reverse iterator for iterating over the rows of a SparseMatrixBase
     * 
     * This iterator inherits from the `Iterator` base class using the CRTP pattern.
     * It is designed to be used with a SparseMatrixBase object to iterate over the rows
     * of the matrix in reverse order. It iterates over the rows in reverse order by
     * starting at the `head->left` entry and moving to the left using the `operator++()`
     * method. It can be indexed to start at any row of the matrix using the `operator[]`
     * method.
     * 
     * @tparam ENTRY_OBJ The entry object class that the iterator will use for its entries.
     */
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


    /**
     * @brief A forward iterator class that iterates over columns of a sparse matrix in a doubly linked list format.
     *
     * This class inherits from the Iterator class and is designed to work with SparseMatrixBase and its
     * subclasses. It is used to iterate over the columns of a sparse matrix in a doubly linked list format.
     *
     * @tparam ENTRY_OBJ The entry object class that the sparse matrix will use for its entries. This class
     * should contain fields for row index, column index, and value.
     */
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

    /**
     * @brief A reverse iterator class that iterates over rows of a sparse matrix in a doubly linked list format.
     *
     * This class inherits from the Iterator class and is designed to work with SparseMatrixBase and its
     * subclasses. It is used to iterate over the rows of a sparse matrix in a doubly linked list format
     * starting from the rightmost element in the row.
     *
     * @tparam ENTRY_OBJ The entry object class that the sparse matrix will use for its entries. This class
     * should contain fields for row index, column index, and value.
     */
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


    /**
     * @brief Returns an iterator that iterates over the given row of the sparse matrix in a forward direction
     * 
     * @param i The row index of the matrix to iterate over
     * @throws invalid_argument If the given index is out of bounds
     * @return RowIterator An iterator object that iterates over the given row
     */
    RowIterator iterate_row(int i){
        if(i<0 || i>=m) throw invalid_argument("Iterator index out of bounds");
        return RowIterator(this)[i];
    }

    /**
     * @brief Returns an iterator that iterates over the given row of the sparse matrix in a reverse direction
     * 
     * @param i The row index of the matrix to iterate over
     * @throws invalid_argument If the given index is out of bounds
     * @return ReverseRowIterator An iterator object that iterates over the given row in reverse
     */
    ReverseRowIterator reverse_iterate_row(int i){
        if(i<0 || i>=m) throw invalid_argument("Iterator index out of bounds");
        return ReverseRowIterator(this)[i];
    }

    /**
     * @brief Returns an iterator that iterates over the given column of the sparse matrix in a forward direction
     * 
     * @param i The column index of the matrix to iterate over
     * @throws invalid_argument If the given index is out of bounds
     * @return ColumnIterator An iterator object that iterates over the given column
     */
    ColumnIterator iterate_column(int i){
        if(i<0 || i>=n) throw invalid_argument("Iterator index out of bounds");
        return ColumnIterator(this)[i];
    }

    /**
     * @brief Returns an iterator that iterates over the given column of the sparse matrix in a reverse direction
     * 
     * @param i The column index of the matrix to iterate over
     * @throws invalid_argument If the given index is out of bounds
     * @return ReverseColumnIterator An iterator object that iterates over the given column in reverse
     */
    ReverseColumnIterator reverse_iterate_column(int i){
        if(i<0 || i>=n) throw invalid_argument("Iterator index out of bounds");
        return ReverseColumnIterator(this)[i];
    }

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

template <class T, template<class> class ENTRY_OBJ=SparseMatrixEntry>
class SparseMatrix: public SparseMatrixBase<ENTRY_OBJ<T>> {
typedef SparseMatrixBase<ENTRY_OBJ<T>> BASE;
public:
    SparseMatrix(int m, int n, int entry_count = 0): BASE::SparseMatrixBase(m,n,entry_count){};
    ENTRY_OBJ<T>* insert_entry(int i, int j, T val = T(1)){
        auto e = BASE::insert_entry(i,j);
        e->value = val;
        return e;
    }

    static shared_ptr<SparseMatrix<T,ENTRY_OBJ>> New(int m, int n, int entry_count = 0){
        return make_shared<SparseMatrix<T,ENTRY_OBJ>>(m,n,entry_count);
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