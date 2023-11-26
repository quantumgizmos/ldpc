#include <iostream>
#include <filesystem>

namespace ldpc{
namespace io{

std::vector<uint8_t> binaryStringToVector(std::string binaryString) {
    std::vector<uint8_t> result;
    for (char c : binaryString) {
        if (c == '0') {
            result.push_back(0);
        } else if (c == '1') {
            result.push_back(1);
        } else {
            throw std::invalid_argument("Input string contains non-binary characters");
        }
    }
    return result;
}

std::vector<std::vector<int>> string_to_csr_vector(std::string str) {
    std::vector<std::vector<int>> result;
    std::stringstream ss(str);
    char c;
    int num;
    while (ss >> c) {
        std::vector<int> row;
        bool left_bracket_observed = false;
        while (ss >> c && c != ']') {
            if (isdigit(c)) {
                ss.putback(c);
                ss >> num;
                row.push_back(num);
            }
            if(c=='[') left_bracket_observed = true;
        }
        if(!left_bracket_observed) break;
        result.push_back(row);
    }
    return result;
}

std::string csr_vector_to_string(const std::vector<std::vector<int>>& vec) {
    std::stringstream ss;
    ss << "[";
    // cout<<vec.size()<<endl;
    for (size_t i = 0; i < vec.size(); i++) {
        ss << "[";
        if(vec[i].size()!=0){
            for (size_t j = 0; j < vec[i].size(); j++) {
                ss << vec[i][j] << ",";
            }
            ss.seekp(-1, std::ios_base::cur); // Remove the trailing comma
        }
        ss << "]";
        if (i != vec.size() - 1) { // Check if this is the last row
            ss << ",";
        }
    }
    ss << "]";
    return ss.str();
}


    std::string getFullPath(const std::string& localPath) {
        // Get the current working directory

        std::string current_file = __FILE__;

        std::filesystem::path project_dir = std::filesystem::path(current_file);

        // Append the local path to the run directory
        std::filesystem::path fullPath = project_dir.parent_path().parent_path() / localPath;

        // Convert the full path to a string and return it
        return fullPath.string();
    }

}//end namespace io
}//end namespace ldpc
