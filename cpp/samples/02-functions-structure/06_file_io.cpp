// 06_file_io.cpp
// Reading and writing files in C++
// Compile: g++ -std=c++17 -o file_io 06_file_io.cpp

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

// ===== Writing to a file =====
void writeTextFile(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }

    file << "Hello, File I/O!" << std::endl;
    file << "This is line 2" << std::endl;
    file << "Numbers: " << 42 << " " << 3.14 << std::endl;

    file.close();
    std::cout << "Successfully wrote to " << filename << std::endl;
}

// ===== Reading from a file =====
void readTextFile(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for reading" << std::endl;
        return;
    }

    std::cout << "\nReading " << filename << ":" << std::endl;
    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << std::endl;
    }

    file.close();
}

// ===== Reading word by word =====
void readWords(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }

    std::cout << "\nReading words from " << filename << ":" << std::endl;
    std::string word;
    while (file >> word) {
        std::cout << "Word: " << word << std::endl;
    }

    file.close();
}

// ===== Writing numbers to file =====
void writeNumbers(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }

    for (float value : data) {
        file << value << "\n";
    }

    file.close();
    std::cout << "Wrote " << data.size() << " numbers to " << filename << std::endl;
}

// ===== Reading numbers from file =====
std::vector<float> readNumbers(const std::string& filename) {
    std::vector<float> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return data;
    }

    float value;
    while (file >> value) {
        data.push_back(value);
    }

    file.close();
    std::cout << "Read " << data.size() << " numbers from " << filename << std::endl;
    return data;
}

// ===== CSV file writing =====
void writeCSV(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }

    // Write header
    file << "x,y,value\n";

    // Write data
    for (int i = 0; i < 5; i++) {
        file << i << "," << i * 2 << "," << i * i << "\n";
    }

    file.close();
    std::cout << "Wrote CSV data to " << filename << std::endl;
}

// ===== CSV file reading =====
void readCSV(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }

    std::cout << "\nReading CSV from " << filename << ":" << std::endl;

    std::string line;
    std::getline(file, line);  // Skip header
    std::cout << "Header: " << line << std::endl;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;

        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }

        if (fields.size() >= 3) {
            std::cout << "x=" << fields[0] << ", y=" << fields[1]
                      << ", value=" << fields[2] << std::endl;
        }
    }

    file.close();
}

// ===== Checking if file exists =====
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// ===== Appending to a file =====
void appendToFile(const std::string& filename, const std::string& text) {
    std::ofstream file(filename, std::ios::app);  // Append mode

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }

    file << text << std::endl;
    file.close();
}

int main() {
    // ===== Basic Writing and Reading =====
    std::cout << "=== Basic File I/O ===" << std::endl;
    writeTextFile("output.txt");
    readTextFile("output.txt");
    readWords("output.txt");

    // ===== Number Data =====
    std::cout << "\n=== Number Data ===" << std::endl;
    std::vector<float> numbers = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    writeNumbers("numbers.txt", numbers);

    std::vector<float> readData = readNumbers("numbers.txt");
    std::cout << "Read numbers: ";
    for (float val : readData) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // ===== CSV Files =====
    std::cout << "\n=== CSV Files ===" << std::endl;
    writeCSV("data.csv");
    readCSV("data.csv");

    // ===== File Existence =====
    std::cout << "\n=== File Existence ===" << std::endl;
    std::cout << "output.txt exists: " << (fileExists("output.txt") ? "Yes" : "No") << std::endl;
    std::cout << "nonexistent.txt exists: " << (fileExists("nonexistent.txt") ? "Yes" : "No") << std::endl;

    // ===== Appending =====
    std::cout << "\n=== Appending ===" << std::endl;
    appendToFile("output.txt", "This line was appended!");
    readTextFile("output.txt");

    return 0;
}

/*
LEARNING NOTES:

FILE STREAMS:
- ifstream: Input file stream (reading)
- ofstream: Output file stream (writing)
- fstream: Both reading and writing

OPENING FILES:
std::ifstream file("filename.txt");
std::ofstream file("filename.txt");
std::ofstream file("filename.txt", std::ios::app);  // Append

FILE MODES:
ios::in      - Read mode
ios::out     - Write mode
ios::app     - Append mode
ios::binary  - Binary mode
ios::trunc   - Truncate file

CHECKING SUCCESS:
if (!file.is_open()) { ... }
if (file.fail()) { ... }
if (file.good()) { ... }

READING METHODS:
file >> variable         - Read word/number
std::getline(file, str)  - Read line
file.read(buffer, size)  - Read binary

WRITING:
file << value            - Write (like cout)
file.write(buffer, size) - Write binary

GOOD PRACTICES:
✓ Check if file opened successfully
✓ Close files when done (or use RAII)
✓ Handle errors gracefully
✓ Use appropriate modes
✓ Binary mode for binary data

GPU RELEVANCE:
- Loading data for GPU processing
- Reading configuration files
- Loading textures/meshes for GPU
- Saving GPU computation results
- Reading training data for ML on GPU

TYPICAL GPU WORKFLOW:
1. Read data from file (CPU)
2. Allocate GPU memory
3. Copy data to GPU
4. Process on GPU
5. Copy results back to CPU
6. Write results to file

BINARY FILES:
For large datasets:
file.write(reinterpret_cast<char*>(data), size * sizeof(float));

CSV FOR GPU DATA:
- Load mesh data (vertices, triangles)
- Load particle positions
- Load image data
- Configuration parameters

TRY THIS:
1. Write a program that reads a list of numbers and computes average
2. Create a log file that appends timestamps
3. Read a configuration file with key=value pairs
4. Write a function to save/load a 2D array
5. Create a binary file writer for float arrays
*/
