/*
 * File: 01_file_io.c
 * Topic: File Input/Output
 *
 * File I/O allows programs to read from and write to files,
 * enabling data persistence beyond program execution.
 *
 * Key Concepts:
 * - Opening and closing files
 * - Reading from files
 * - Writing to files
 * - File modes
 * - Error handling
 * - Text vs binary files
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Writing to a file
    printf("=== Writing to File ===\n");

    FILE *write_file = fopen("output.txt", "w");

    if (write_file == NULL) {
        printf("Error: Could not open file for writing\n");
        return 1;
    }

    fprintf(write_file, "Hello, File I/O!\n");
    fprintf(write_file, "This is line 2\n");
    fprintf(write_file, "Numbers: %d, %.2f\n", 42, 3.14);

    fclose(write_file);
    printf("Data written to output.txt\n");

    // Reading from a file
    printf("\n=== Reading from File ===\n");

    FILE *read_file = fopen("output.txt", "r");

    if (read_file == NULL) {
        printf("Error: Could not open file for reading\n");
        return 1;
    }

    char line[100];
    printf("File contents:\n");
    while (fgets(line, sizeof(line), read_file) != NULL) {
        printf("%s", line);
    }

    fclose(read_file);

    // Appending to a file
    printf("\n=== Appending to File ===\n");

    FILE *append_file = fopen("output.txt", "a");

    if (append_file == NULL) {
        printf("Error: Could not open file for appending\n");
        return 1;
    }

    fprintf(append_file, "This line was appended\n");
    fclose(append_file);
    printf("Data appended to output.txt\n");

    // Reading character by character
    printf("\n=== Reading Characters ===\n");

    read_file = fopen("output.txt", "r");

    if (read_file == NULL) {
        printf("Error: Could not open file\n");
        return 1;
    }

    printf("First 50 characters:\n");
    int ch;
    int count = 0;
    while ((ch = fgetc(read_file)) != EOF && count < 50) {
        putchar(ch);
        count++;
    }
    printf("\n");

    fclose(read_file);

    // Formatted reading
    printf("\n=== Formatted Reading ===\n");

    // Create a data file
    FILE *data_file = fopen("data.txt", "w");
    fprintf(data_file, "Alice 25 3.8\n");
    fprintf(data_file, "Bob 22 3.5\n");
    fprintf(data_file, "Charlie 24 3.9\n");
    fclose(data_file);

    // Read formatted data
    data_file = fopen("data.txt", "r");

    if (data_file == NULL) {
        printf("Error: Could not open data file\n");
        return 1;
    }

    printf("Student records:\n");
    char name[50];
    int age;
    float gpa;

    while (fscanf(data_file, "%s %d %f", name, &age, &gpa) == 3) {
        printf("Name: %-10s Age: %d  GPA: %.1f\n", name, age, gpa);
    }

    fclose(data_file);

    // File position
    printf("\n=== File Positioning ===\n");

    FILE *pos_file = fopen("data.txt", "r");

    if (pos_file == NULL) {
        printf("Error: Could not open file\n");
        return 1;
    }

    printf("Reading first record:\n");
    fscanf(pos_file, "%s %d %f", name, &age, &gpa);
    printf("%s %d %.1f\n", name, age, gpa);

    printf("Current position: %ld\n", ftell(pos_file));

    // Rewind to start
    rewind(pos_file);
    printf("After rewind, position: %ld\n", ftell(pos_file));

    // Seek to position
    fseek(pos_file, 0, SEEK_END);  // Go to end
    printf("File size: %ld bytes\n", ftell(pos_file));

    fclose(pos_file);

    // Binary file I/O
    printf("\n=== Binary File I/O ===\n");

    int numbers[] = {10, 20, 30, 40, 50};
    int read_numbers[5];

    // Write binary
    FILE *bin_file = fopen("numbers.bin", "wb");

    if (bin_file == NULL) {
        printf("Error: Could not open binary file for writing\n");
        return 1;
    }

    fwrite(numbers, sizeof(int), 5, bin_file);
    fclose(bin_file);
    printf("Binary data written\n");

    // Read binary
    bin_file = fopen("numbers.bin", "rb");

    if (bin_file == NULL) {
        printf("Error: Could not open binary file for reading\n");
        return 1;
    }

    fread(read_numbers, sizeof(int), 5, bin_file);
    fclose(bin_file);

    printf("Numbers read from binary file: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", read_numbers[i]);
    }
    printf("\n");

    // File existence check
    printf("\n=== Checking File Existence ===\n");

    FILE *check = fopen("nonexistent.txt", "r");
    if (check == NULL) {
        printf("File does not exist\n");
    } else {
        printf("File exists\n");
        fclose(check);
    }

    // Error handling with perror
    printf("\n=== Error Handling ===\n");
    FILE *error_file = fopen("/invalid/path/file.txt", "r");
    if (error_file == NULL) {
        perror("Error opening file");
    }

    printf("\n=== Cleanup ===\n");
    printf("Removing temporary files...\n");
    remove("output.txt");
    remove("data.txt");
    remove("numbers.bin");
    printf("Done!\n");

    return 0;
}

/*
 * FILE MODES:
 * "r"  - Read (file must exist)
 * "w"  - Write (creates new or truncates existing)
 * "a"  - Append (creates new or appends to existing)
 * "r+" - Read and write (file must exist)
 * "w+" - Read and write (creates new or truncates)
 * "a+" - Read and append
 *
 * Binary modes: add 'b' (e.g., "rb", "wb", "ab")
 *
 * KEY FUNCTIONS:
 * fopen(filename, mode)    - Open file
 * fclose(file)             - Close file
 * fprintf(file, format...) - Write formatted
 * fscanf(file, format...)  - Read formatted
 * fgets(buffer, size, file)- Read line
 * fputs(string, file)      - Write string
 * fgetc(file)              - Read character
 * fputc(char, file)        - Write character
 * fread(ptr, size, count, file)  - Read binary
 * fwrite(ptr, size, count, file) - Write binary
 * fseek(file, offset, whence)    - Set position
 * ftell(file)              - Get position
 * rewind(file)             - Reset to start
 *
 * ERROR HANDLING:
 * - Always check if fopen() returns NULL
 * - Check return values of read/write operations
 * - Use perror() to print error messages
 * - Always close files with fclose()
 *
 * BEST PRACTICES:
 * 1. Always check if file opened successfully
 * 2. Always close files when done
 * 3. Use appropriate file mode
 * 4. Handle errors gracefully
 * 5. Check return values
 *
 * EXERCISES:
 * 1. Copy one file to another
 * 2. Count lines, words, and characters in a file
 * 3. Merge two files into a third file
 * 4. Read CSV file and parse data
 * 5. Create a simple file encryption program
 * 6. Build a text file search program
 * 7. Implement a log file system
 * 8. Create a simple database using files
 */
