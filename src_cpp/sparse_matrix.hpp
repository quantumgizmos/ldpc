#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <memory>
#include <iterator>

using namespace std;

// A container for each matrix element
template <class DERIVED>
class entry_base {
    public:
        int row_index=-100; //row index
        int col_index=-100; //col_index index
        
        DERIVED* left = static_cast<DERIVED*>(this); //pointer to previous row element
        DERIVED* right = static_cast<DERIVED*>(this); //pointer to next row element
        DERIVED* up = static_cast<DERIVED*>(this); //pointer to previous column element
        DERIVED* down = static_cast<DERIVED*>(this); //pointer to next column element
       
        void reset(){
            row_index=-100;
            col_index=-100;
            left = static_cast<DERIVED*>(this);
            right = static_cast<DERIVED*>(this);
            up = static_cast<DERIVED*>(this);
            down = static_cast<DERIVED*>(this);
        }

        bool at_end(){
            if(row_index==-100) return true;
            else return false;
        }

    ~entry_base(){};
};


template <class T>
class sparse_matrix_entry: public entry_base<sparse_matrix_entry<T>> {
    public:
        T value = T(0); // the value structure we are storing at each matrix location. We can define this as any object, and overload operators.
    ~sparse_matrix_entry(){};
};

template <class ENTRY_OBJ>
class sparse_matrix_base {
  public:
    int m,n; //m: check-count; n: bit-count
    int node_count;
    int entry_block_size, released_entry_count;
    vector<ENTRY_OBJ> entries;
    vector<ENTRY_OBJ*> removed_entries;       
    vector<ENTRY_OBJ*> row_heads; //starting point for each row
    vector<ENTRY_OBJ*> column_heads; //starting point for each column

    // vector<ENTRY_OBJ*> matrix_entries;
    
    sparse_matrix_base(int const check_count, int bit_count){
        m=check_count;
        n=bit_count;
        this->entry_block_size = int(m+n);
        this->released_entry_count=0;
        allocate();
    }

    ~sparse_matrix_base(){
        this->entries.clear();
    }

    ENTRY_OBJ* allocate_new_entry(){
        if(this->removed_entries.size()!=0){
            auto e = this->removed_entries.back();
            this->removed_entries.pop_back();
            return e; 
        }
        if(this->released_entry_count==this->entries.size()){
            this->entries.resize(this->entries.size()+this->entry_block_size);
        }
        auto e = &this->entries[released_entry_count];
        released_entry_count++;
        return e;
    }

    int entry_count(){
        return this->released_entry_count - this->n - this->m - this->removed_entries.size();
    }


    //this function allocates space for an mxn matrix.
    void allocate(){
        
        row_heads.resize(m);
        column_heads.resize(n);

        for(int i=0;i<m;i++){
            ENTRY_OBJ* row_entry;
            row_entry = this->allocate_new_entry();
            row_entry->row_index = -100; //we set the row head index to -100 to indicate is not a value element
            row_entry->col_index = -100;
            row_entry->right = row_entry; //at first the head elements point to themselves
            row_entry->left = row_entry;
            row_entry->up = row_entry;
            row_entry->down = row_entry;
            row_heads[i]=row_entry;
        }

        for(int i=0;i<n;i++){
            ENTRY_OBJ* column_entry; 
            column_entry = this->allocate_new_entry();
            column_entry->row_index = -100;
            column_entry->col_index = -100;
            column_entry->right = column_entry;
            column_entry->left = column_entry;
            column_entry->up = column_entry;
            column_entry->down = column_entry;
            column_heads[i]=column_entry;
        }
    }

    void swap_rows(int i, int j){
        auto tmp1 = row_heads[i];
        auto tmp2 = row_heads[j];
        row_heads[j] = tmp1;
        row_heads[i] = tmp2;
        for(auto e: iterate_row(i)) e->row_index=i;
        for(auto e: iterate_row(j)) e->row_index=j;
    }

    void reorder_rows(vector<int> rows){

        vector<ENTRY_OBJ*> temp_row_heads;
        for(int i = 0; i<m; i++) temp_row_heads.push_back(row_heads[i]);
        for(int i = 0; i<m; i++){
            row_heads[i] = temp_row_heads[rows[i]];
            for(auto e: iterate_row(i)){
                e->row_index = i;
            }
        }

    }

    void swap_columns(int i, int j){
        auto tmp1 = column_heads[i];
        auto tmp2 = column_heads[j];
        column_heads[j] = tmp1;
        column_heads[i] = tmp2;
        for(auto e: iterate_column(i)) e->col_index=i;
        for(auto e: iterate_column(j)) e->col_index=j;
    }

    void remove_entry(int i, int j){
        auto e = get_entry(i,j);
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
            e->reset();
            this->removed_entries.push_back(e);
        }
    }

    ENTRY_OBJ* insert_entry(int j, int i){
        if(j>=m || i>=n || j<0 || i<0) throw invalid_argument("Index i or j is out of bounds"); 
        ENTRY_OBJ* entry;
        ENTRY_OBJ* right_entry;
        ENTRY_OBJ* left_entry;
        ENTRY_OBJ* up_entry;
        ENTRY_OBJ* down_entry;
        
        entry = row_heads[j];
        left_entry = row_heads[j];
        right_entry = row_heads[j];
        
        entry=entry->right;
        for(entry; entry!=row_heads[j];entry=entry->right){
            
            if(entry->col_index == i){
                return entry;
            }
            
            if(entry->col_index < i) left_entry = entry;
            
            if(entry->col_index > i) {
                right_entry = entry;
                break;
            }
        
        }

        entry = column_heads[i];
        up_entry = column_heads[i];
        down_entry = column_heads[i];

        entry = entry->down;
        for(entry; entry!=column_heads[i];entry=entry->down){
            if(entry->row_index < j) up_entry = entry;
            if(entry->row_index > j) {
                down_entry = entry;
                break;
            }
        }

        ENTRY_OBJ* e;
        e = this->allocate_new_entry();
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

        // matrix_entries.push_back(e);
        return e;     
    }

    ENTRY_OBJ* get_entry(int j, int i){
        if(j>=m || i>=n || j<0 || i<0) throw invalid_argument("Index i or j is out of bounds"); 
        ENTRY_OBJ* e;
        e = column_heads[i]->down;
        for(e;e!=column_heads[i];e=e->down){
            if(e->row_index==j) return e;
        }
        return column_heads[i];
        
        }



    template <class DERIVED>
    class Iterator{
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            sparse_matrix_base<ENTRY_OBJ>* matrix;
            ENTRY_OBJ* e;
            ENTRY_OBJ* head;
            // int index = -1;
            Iterator(sparse_matrix_base<ENTRY_OBJ>* mat){
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
            RowIterator(sparse_matrix_base<ENTRY_OBJ>* mat) : BASE::Iterator(mat){}

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
            ReverseRowIterator(sparse_matrix_base<ENTRY_OBJ>* mat) : BASE::Iterator(mat){}

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
            ColumnIterator(sparse_matrix_base<ENTRY_OBJ>* mat) : BASE::Iterator(mat){}

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
            ReverseColumnIterator(sparse_matrix_base<ENTRY_OBJ>* mat) : BASE::Iterator(mat){}

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


template <class T, template<class> class ENTRY_OBJ=sparse_matrix_entry>
class sparse_matrix: public sparse_matrix_base<ENTRY_OBJ<T>> {
  typedef sparse_matrix_base<ENTRY_OBJ<T>> BASE;
  public:
    sparse_matrix(int m, int n): BASE::sparse_matrix_base(m,n){};
    ENTRY_OBJ<T>* insert_entry(int i, int j, T val = T(1)){
        auto e = BASE::insert_entry(i,j);
        e->value = val;
        return e;
    }

    ~sparse_matrix(){};
};


#endif