template<class ENTRY_OBJ>
class EntryMemoryManager{
    public:
        int entry_block_size;
        int released_entry_count;
        int current_entry_block;
        vector<vector<ENTRY_OBJ>> entries;
        vector<ENTRY_OBJ*> removed_entries;

        EntryMemoryManager(int initial_entry_count, int entry_block_size){

            this->released_entry_count = 0;
            this->current_entry_block = 0;
            this->entry_block_size = entry_block_size;
            vector<ENTRY_OBJ> initial_block;
            initial_block.resize(initial_entry_count);
            entries.push_back(initial_block);

        }

        ~EntryMemoryManager(){
            entries.clear();
        }

        ENTRY_OBJ* get_new_entry(){

            if(this->removed_entries.size()!=0){
                auto e = this->removed_entries.back();
                this->removed_entries.pop_back();
                return e; 
            }
            if(this->released_entry_count==this->entries[this->current_entry_block].size()){
                vector<ENTRY_OBJ> new_entry_block;
                new_entry_block.resize(this->entry_block_size);
                
                
                // int new_size = this->entries.size()+this->entry_block_size;
                // // this->entries.resize(new_size);
                // for(int i = 0; i<this->entry_block_size; i++){
                //     auto e = new ENTRY_OBJ();
                //     this->entries.push_back(e);
                // }
            }

        
            auto e = this->entries[this->current_entry_block][this->released_entry_count];
            this->released_entry_count++;

            return e;
        }

};