/*
 * File: 06_strings.c
 * Topic: Strings in C
 *
 * In C, strings are arrays of characters terminated by a null
 * character '\0'. Understanding strings is crucial for text processing.
 *
 * Key Concepts:
 * - String declaration and initialization
 * - String input/output
 * - Null terminator
 * - Character arrays vs string literals
 * - Basic string operations
 */

#include <stdio.h>
#include <string.h>  // String library functions

int main() {
    // String declaration and initialization
    printf("=== String Declaration ===\n");

    // Method 1: Character array with explicit null terminator
    char str1[] = {'H', 'e', 'l', 'l', 'o', '\0'};

    // Method 2: String literal (automatic null terminator)
    char str2[] = "World";

    // Method 3: Specify size
    char str3[50] = "C Programming";

    printf("str1: %s\n", str1);
    printf("str2: %s\n", str2);
    printf("str3: %s\n", str3);

    // Understanding null terminator
    printf("\n=== Null Terminator ===\n");
    char demo[] = "ABC";
    printf("Characters: ");
    for (int i = 0; i < 4; i++) {  // Including '\0'
        if (demo[i] == '\0') {
            printf("\\0");
        } else {
            printf("%c ", demo[i]);
        }
    }
    printf("\n");
    printf("String length: %zu (excluding null terminator)\n", strlen(demo));

    // String input
    printf("\n=== String Input ===\n");
    char name[50];
    printf("Enter your name: ");
    scanf("%s", name);  // Reads until whitespace (no & needed!)
    printf("Hello, %s!\n", name);

    // Reading line with spaces (using fgets)
    printf("\nEnter full name with spaces: ");
    getchar();  // Consume newline from previous scanf
    fgets(name, sizeof(name), stdin);
    name[strcspn(name, "\n")] = '\0';  // Remove trailing newline
    printf("Welcome, %s!\n", name);

    // String length
    printf("\n=== String Length ===\n");
    char message[] = "Hello, World!";
    printf("String: %s\n", message);
    printf("Length: %zu characters\n", strlen(message));

    // String copy
    printf("\n=== String Copy ===\n");
    char source[] = "Copy me";
    char destination[50];
    strcpy(destination, source);
    printf("Source: %s\n", source);
    printf("Destination: %s\n", destination);

    // String concatenation
    printf("\n=== String Concatenation ===\n");
    char first[50] = "Hello, ";
    char second[] = "World!";
    strcat(first, second);  // Appends second to first
    printf("Result: %s\n", first);

    // String comparison
    printf("\n=== String Comparison ===\n");
    char str_a[] = "Apple";
    char str_b[] = "Banana";
    char str_c[] = "Apple";

    int cmp1 = strcmp(str_a, str_b);
    int cmp2 = strcmp(str_a, str_c);

    printf("strcmp(\"%s\", \"%s\") = %d ", str_a, str_b, cmp1);
    if (cmp1 < 0) printf("(Apple comes before Banana)\n");

    printf("strcmp(\"%s\", \"%s\") = %d ", str_a, str_c, cmp2);
    if (cmp2 == 0) printf("(Strings are equal)\n");

    // Character access and modification
    printf("\n=== Character Access ===\n");
    char word[] = "Hello";
    printf("Original: %s\n", word);
    word[0] = 'h';  // Change first character
    printf("Modified: %s\n", word);

    // Iterating through string
    printf("\nCharacters: ");
    for (int i = 0; word[i] != '\0'; i++) {
        printf("%c ", word[i]);
    }
    printf("\n");

    // String search
    printf("\n=== String Search ===\n");
    char text[] = "The quick brown fox";
    char *found = strstr(text, "brown");
    if (found != NULL) {
        printf("Found 'brown' at position: %ld\n", found - text);
    }

    // Finding character in string
    char *char_pos = strchr(text, 'q');
    if (char_pos != NULL) {
        printf("Found 'q' at position: %ld\n", char_pos - text);
    }

    // Converting case (manual)
    printf("\n=== Case Conversion ===\n");
    char lower[] = "hello world";
    char upper[50];

    for (int i = 0; lower[i] != '\0'; i++) {
        if (lower[i] >= 'a' && lower[i] <= 'z') {
            upper[i] = lower[i] - 32;  // Convert to uppercase
        } else {
            upper[i] = lower[i];
        }
    }
    upper[strlen(lower)] = '\0';

    printf("Lowercase: %s\n", lower);
    printf("Uppercase: %s\n", upper);

    // Array of strings
    printf("\n=== Array of Strings ===\n");
    char days[7][10] = {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    };

    printf("Days of the week:\n");
    for (int i = 0; i < 7; i++) {
        printf("%d. %s\n", i + 1, days[i]);
    }

    return 0;
}

/*
 * IMPORTANT STRING FUNCTIONS (string.h):
 * strlen(str)         - Get string length
 * strcpy(dest, src)   - Copy string
 * strncpy(dest, src, n) - Copy n characters
 * strcat(dest, src)   - Concatenate strings
 * strcmp(str1, str2)  - Compare strings (returns 0 if equal)
 * strchr(str, ch)     - Find character in string
 * strstr(str, substr) - Find substring
 *
 * IMPORTANT NOTES:
 * - Strings must have null terminator '\0'
 * - Always ensure enough space for strings
 * - scanf() with %s stops at whitespace
 * - Use fgets() for strings with spaces
 * - String literals are stored in read-only memory
 *
 * COMMON MISTAKES:
 * - Not allocating enough space for null terminator
 * - Buffer overflow with strcpy/strcat
 * - Forgetting & is not needed for strings in scanf
 *
 * EXERCISES:
 * 1. Write a function to count vowels in a string
 * 2. Reverse a string
 * 3. Check if a string is a palindrome
 * 4. Count words in a string
 * 5. Remove all spaces from a string
 * 6. Convert string to uppercase/lowercase
 * 7. Find the longest word in a sentence
 * 8. Check if two strings are anagrams
 */
