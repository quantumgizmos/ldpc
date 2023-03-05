template<class ENTRY_OBJ>
class EntryMemoryManager{
    public:
        int entry_block_size;
        int released_entry_count;
        int current_entry_block;
        int allocated_entry_count;
        vector<vector<ENTRY_OBJ>> entries;
        vector<ENTRY_OBJ*> removed_entries;

        EntryMemoryManager(int initial_entry_count, int entry_block_size){

            this->released_entry_count = 0;
            this->current_entry_block = 0;
            this->entry_block_size = entry_block_size;
            vector<ENTRY_OBJ> initial_block;
            initial_block.resize(initial_entry_count);
            this->allocated_entry_count+=initial_entry_count;
            entries.push_back(initial_block);

        }

        ~EntryMemoryManager(){
            for(auto block: this->entries) block.clear();
            this->entries.clear();
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
                this->allocated_entry_count+=this->entry_block_size;
                entries.push_back(new_entry_block);
            }

        
            auto e = &this->entries[this->current_entry_block][this->released_entry_count];
            this->released_entry_count++;

            return e;
        }

};