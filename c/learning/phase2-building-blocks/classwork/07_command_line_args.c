/*
 * File: 07_command_line_args.c
 * Topic: Command-Line Arguments
 *
 * Command-line arguments allow programs to receive input when they start.
 * This is how most command-line tools work (ls, grep, git, etc.)
 *
 * Key Concepts:
 * - argc (argument count)
 * - argv (argument vector/array)
 * - Processing command-line arguments
 * - Parsing options and flags
 * - Practical applications
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function prototypes
void printUsage(char *program_name);
void demonstrateBasics(int argc, char *argv[]);
void simpleCalculator(int argc, char *argv[]);
void fileProcessor(int argc, char *argv[]);

int main(int argc, char *argv[]) {
    /*
     * argc: Argument Count - number of command-line arguments
     * argv: Argument Vector - array of strings (char pointers)
     *       argv[0] is always the program name
     *       argv[1] to argv[argc-1] are the actual arguments
     *       argv[argc] is NULL
     */

    printf("=== Basic Command-Line Arguments ===\n");
    demonstrateBasics(argc, argv);

    // If no arguments provided, show usage
    if (argc == 1) {
        printf("\n");
        printUsage(argv[0]);
        printf("\nExamples:\n");
        printf("  %s calc 10 + 20\n", argv[0]);
        printf("  %s info\n", argv[0]);
        printf("  %s file input.txt\n", argv[0]);
        return 0;
    }

    // Parse commands
    if (strcmp(argv[1], "calc") == 0) {
        simpleCalculator(argc, argv);
    }
    else if (strcmp(argv[1], "file") == 0) {
        fileProcessor(argc, argv);
    }
    else if (strcmp(argv[1], "info") == 0) {
        printf("\n=== Program Information ===\n");
        printf("This program demonstrates command-line argument handling.\n");
        printf("Version: 1.0\n");
    }
    else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        printUsage(argv[0]);
    }
    else {
        printf("\nUnknown command: %s\n", argv[1]);
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}

// Print usage information
void printUsage(char *program_name) {
    printf("\n=== Usage ===\n");
    printf("  %s <command> [arguments]\n\n", program_name);
    printf("Commands:\n");
    printf("  calc <num1> <op> <num2>  - Perform calculation\n");
    printf("  file <filename>           - Process file\n");
    printf("  info                      - Show program info\n");
    printf("  -h, --help                - Show this help\n");
}

// Demonstrate basic argc and argv
void demonstrateBasics(int argc, char *argv[]) {
    printf("Number of arguments (argc): %d\n", argc);
    printf("\nAll arguments:\n");

    for (int i = 0; i < argc; i++) {
        printf("  argv[%d] = \"%s\"\n", i, argv[i]);
    }

    if (argc > 1) {
        printf("\nProgram name: %s\n", argv[0]);
        printf("First argument: %s\n", argv[1]);
    }
}

// Simple calculator using command-line arguments
void simpleCalculator(int argc, char *argv[]) {
    printf("\n=== Calculator Mode ===\n");

    if (argc != 5) {
        printf("Error: Calculator needs 3 arguments\n");
        printf("Usage: %s calc <num1> <operator> <num2>\n", argv[0]);
        printf("Example: %s calc 10 + 20\n", argv[0]);
        return;
    }

    // Parse arguments
    double num1 = atof(argv[2]);  // Convert string to double
    char op = argv[3][0];         // Get operator character
    double num2 = atof(argv[4]);

    printf("Calculating: %.2f %c %.2f\n", num1, op, num2);

    double result;
    int valid = 1;

    switch (op) {
        case '+':
            result = num1 + num2;
            break;
        case '-':
            result = num1 - num2;
            break;
        case 'x':
        case '*':
            result = num1 * num2;
            break;
        case '/':
            if (num2 != 0) {
                result = num1 / num2;
            } else {
                printf("Error: Division by zero!\n");
                valid = 0;
            }
            break;
        default:
            printf("Error: Unknown operator '%c'\n", op);
            printf("Supported: +, -, *, x, /\n");
            valid = 0;
    }

    if (valid) {
        printf("Result: %.2f\n", result);
    }
}

// File processor demonstration
void fileProcessor(int argc, char *argv[]) {
    printf("\n=== File Processor Mode ===\n");

    if (argc < 3) {
        printf("Error: No filename provided\n");
        printf("Usage: %s file <filename> [options]\n", argv[0]);
        return;
    }

    char *filename = argv[2];
    printf("Processing file: %s\n", filename);

    // Check for optional flags
    int verbose = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    if (verbose) {
        printf("Verbose mode enabled\n");
    }

    // Try to open and process file
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("Error: Cannot open file '%s'\n", filename);
        return;
    }

    printf("File opened successfully!\n");

    // Count lines
    int lines = 0;
    int chars = 0;
    int ch;

    while ((ch = fgetc(file)) != EOF) {
        chars++;
        if (ch == '\n') {
            lines++;
        }
    }

    printf("Lines: %d\n", lines);
    printf("Characters: %d\n", chars);

    fclose(file);
}

/*
 * RUNNING THIS PROGRAM:
 *
 * Compile:
 *   gcc 07_command_line_args.c -o cmdargs
 *
 * Run without arguments:
 *   ./cmdargs
 *
 * Run with arguments:
 *   ./cmdargs calc 10 + 20
 *   ./cmdargs calc 15 - 5
 *   ./cmdargs calc 6 x 7
 *   ./cmdargs info
 *   ./cmdargs -h
 *   ./cmdargs file sample.txt
 *   ./cmdargs file sample.txt -v
 *
 * IMPORTANT NOTES:
 *
 * 1. argc is always >= 1 (program name is argv[0])
 * 2. argv is an array of strings (char*)
 * 3. argv[argc] is NULL (marks the end)
 * 4. Arguments are always strings - use atoi(), atof() to convert
 * 5. Shell may interpret special characters (* , >, etc.)
 * 6. Use quotes for arguments with spaces: "./prog "hello world""
 *
 * ARGUMENT PARSING PATTERNS:
 *
 * Short options: -h, -v, -a
 * Long options: --help, --verbose, --all
 * Options with values: -o output.txt, --output=output.txt
 * Multiple short options: -abc (same as -a -b -c)
 * Flags: present or absent (boolean)
 * Values: require an argument
 *
 * COMMON FUNCTIONS:
 *
 * atoi(str)    - Convert string to int
 * atof(str)    - Convert string to float/double
 * atol(str)    - Convert string to long
 * strtol()     - Convert with error checking
 * strcmp()     - Compare strings
 * getopt()     - Library function for parsing (not covered here)
 *
 * BEST PRACTICES:
 *
 * 1. Always check argc before accessing argv[i]
 * 2. Provide help text (-h, --help)
 * 3. Validate input arguments
 * 4. Give clear error messages
 * 5. Return proper exit codes (0 = success, non-zero = error)
 * 6. Handle edge cases (no args, too many args)
 * 7. Use consistent naming for options
 *
 * REAL-WORLD EXAMPLES:
 *
 * ls -la                  // List files (long format, all)
 * gcc -o output file.c    // Compile with output name
 * grep -r "text" dir/     // Recursive search
 * git commit -m "msg"     // Git with message
 * curl -X POST url        // HTTP request with method
 *
 * EXERCISES:
 *
 * 1. Create a grep-like tool that searches for text in files
 * 2. Build a file converter (e.g., CSV to JSON)
 * 3. Make a todo list program with add/list/delete commands
 * 4. Implement a simple HTTP client with various options
 * 5. Create a calculator that supports multiple operations
 * 6. Build a file encryption tool with password from command line
 * 7. Write a batch file renamer
 * 8. Create a log analyzer with filtering options
 * 9. Implement a unit converter (temperature, length, etc.)
 * 10. Build a simple database query tool
 *
 * ADVANCED TOPICS:
 *
 * - getopt() and getopt_long() for complex parsing
 * - Environment variables (getenv())
 * - Configuration files vs command-line args
 * - Interactive vs batch mode
 * - stdin/stdout redirection
 * - Pipes and process communication
 */
