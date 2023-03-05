#include <iostream>
#include <filesystem>

namespace io{


std::vector<std::vector<int>> string_to_csr_vector(std::string str) {
    std::vector<std::vector<int>> result;
    std::stringstream ss(str);
    char c;
    int num;
    while (ss >> c) {
        std::vector<int> row;
        bool left_bracket_observed = false;
        while (ss >> c && c != '}') {
            if (isdigit(c)) {
                ss.putback(c);
                ss >> num;
                row.push_back(num);
            }
            if(c=='{') left_bracket_observed = true;
        }
        if(!left_bracket_observed) break;
        result.push_back(row);
    }
    return result;
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



    // std::string getFullPath(const std::string& local_path) {
    //     std::filesystem::path current_dir = std::filesystem::current_path();
    //     std::filesystem::path cmake_file = "CMakeLists.txt";

    //     // Search for the CMakeLists.txt file
    //     while (!std::filesystem::exists(current_dir / cmake_file)) {
    //         // If we reach the root directory, exit the loop
    //         if (current_dir == current_dir.root_path()) {
    //             break;
    //         }
    //         // Go up one directory
    //         current_dir = current_dir.parent_path();
    //     }

    //     // Concatenate the local path to the CMake directory
    //     std::filesystem::path full_path = current_dir / local_path;

    //     std::cout<<full_path.string()<<std::endl;

    //     // Return the full path as a string, or an empty string if not found
    //     return std::filesystem::exists(full_path) ? full_path.string() : "";
    // }

}