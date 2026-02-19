/*
 * File: 02_text_analyzer.c
 * Topic: Text File Analyzer
 *
 * A practical project that analyzes text files and provides statistics.
 *
 * Concepts demonstrated:
 * - File I/O
 * - String processing
 * - Character analysis
 * - Data structures
 * - Functions
 *
 * Features:
 * - Count lines, words, characters
 * - Find most frequent word
 * - Count vowels and consonants
 * - Calculate average word length
 * - Display file statistics
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 1000
#define MAX_WORD 100
#define MAX_WORDS 1000

// Word frequency structure
typedef struct {
    char word[MAX_WORD];
    int count;
} WordFreq;

// Text statistics structure
typedef struct {
    int lines;
    int words;
    int characters;
    int vowels;
    int consonants;
    int digits;
    int spaces;
    int sentences;
} TextStats;

// Function prototypes
void analyzeFile(const char* filename);
void countBasicStats(FILE* file, TextStats* stats);
void findMostFrequentWord(FILE* file);
void displayStatistics(TextStats* stats);
int isVowel(char c);
void toLowerCase(char* str);
void createSampleFile(void);

int main() {
    char filename[100];
    int choice;

    printf("=====================================\n");
    printf("     Text File Analyzer\n");
    printf("=====================================\n\n");

    while (1) {
        printf("\n1. Analyze a file\n");
        printf("2. Create sample file for testing\n");
        printf("3. Exit\n");
        printf("\nEnter your choice: ");

        if (scanf("%d", &choice) != 1) {
            printf("Invalid input!\n");
            while (getchar() != '\n');
            continue;
        }
        while (getchar() != '\n');  // Clear buffer

        switch (choice) {
            case 1:
                printf("\nEnter filename: ");
                fgets(filename, sizeof(filename), stdin);
                filename[strcspn(filename, "\n")] = '\0';

                analyzeFile(filename);
                break;

            case 2:
                createSampleFile();
                printf("\nSample file 'sample.txt' created successfully!\n");
                printf("You can now analyze it.\n");
                break;

            case 3:
                printf("\nGoodbye!\n");
                return 0;

            default:
                printf("Invalid choice!\n");
        }
    }

    return 0;
}

// Main analysis function
void analyzeFile(const char* filename) {
    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        printf("\nError: Could not open file '%s'\n", filename);
        printf("Make sure the file exists and try again.\n");
        return;
    }

    printf("\n=====================================\n");
    printf("Analyzing file: %s\n", filename);
    printf("=====================================\n");

    TextStats stats = {0};

    // Count basic statistics
    countBasicStats(file, &stats);
    fclose(file);

    // Display statistics
    displayStatistics(&stats);

    // Find most frequent word
    file = fopen(filename, "r");
    findMostFrequentWord(file);
    fclose(file);
}

// Count lines, words, characters, etc.
void countBasicStats(FILE* file, TextStats* stats) {
    char line[MAX_LINE];
    int in_word = 0;

    while (fgets(line, sizeof(line), file) != NULL) {
        stats->lines++;

        for (int i = 0; line[i] != '\0'; i++) {
            char c = line[i];
            stats->characters++;

            if (c == '.' || c == '!' || c == '?') {
                stats->sentences++;
            }

            if (isspace(c)) {
                stats->spaces++;
                in_word = 0;
            } else {
                if (!in_word) {
                    stats->words++;
                    in_word = 1;
                }

                if (isalpha(c)) {
                    if (isVowel(tolower(c))) {
                        stats->vowels++;
                    } else {
                        stats->consonants++;
                    }
                } else if (isdigit(c)) {
                    stats->digits++;
                }
            }
        }
        in_word = 0;  // Reset at end of line
    }
}

// Find and display most frequent word
void findMostFrequentWord(FILE* file) {
    WordFreq words[MAX_WORDS] = {0};
    int word_count = 0;
    char word[MAX_WORD];
    int ch;
    int word_index = 0;

    printf("\n--- Word Frequency Analysis ---\n");

    rewind(file);  // Go back to start of file

    // Read word by word
    while ((ch = fgetc(file)) != EOF) {
        if (isalnum(ch)) {
            if (word_index < MAX_WORD - 1) {
                word[word_index++] = ch;
            }
        } else if (word_index > 0) {
            word[word_index] = '\0';
            toLowerCase(word);

            // Skip very short words
            if (strlen(word) < 2) {
                word_index = 0;
                continue;
            }

            // Find or add word
            int found = 0;
            for (int i = 0; i < word_count; i++) {
                if (strcmp(words[i].word, word) == 0) {
                    words[i].count++;
                    found = 1;
                    break;
                }
            }

            if (!found && word_count < MAX_WORDS) {
                strcpy(words[word_count].word, word);
                words[word_count].count = 1;
                word_count++;
            }

            word_index = 0;
        }
    }

    if (word_count == 0) {
        printf("No words found in file.\n");
        return;
    }

    // Find most frequent
    int max_count = 0;
    int max_index = 0;

    for (int i = 0; i < word_count; i++) {
        if (words[i].count > max_count) {
            max_count = words[i].count;
            max_index = i;
        }
    }

    printf("Most frequent word: '%s' (appears %d times)\n",
           words[max_index].word, max_count);

    // Display top 5 words
    printf("\nTop 5 most frequent words:\n");
    for (int rank = 0; rank < 5 && rank < word_count; rank++) {
        max_count = 0;
        max_index = -1;

        for (int i = 0; i < word_count; i++) {
            if (words[i].count > max_count) {
                // Check if not already displayed
                int already_shown = 0;
                for (int j = 0; j < rank; j++) {
                    if (words[i].count == 0) {
                        already_shown = 1;
                        break;
                    }
                }

                if (!already_shown) {
                    max_count = words[i].count;
                    max_index = i;
                }
            }
        }

        if (max_index != -1) {
            printf("%d. '%s' - %d occurrences\n",
                   rank + 1, words[max_index].word, max_count);
            words[max_index].count = 0;  // Mark as displayed
        }
    }
}

// Display all statistics
void displayStatistics(TextStats* stats) {
    printf("\n--- File Statistics ---\n");
    printf("Lines:       %d\n", stats->lines);
    printf("Words:       %d\n", stats->words);
    printf("Characters:  %d\n", stats->characters);
    printf("Sentences:   %d\n", stats->sentences);
    printf("Vowels:      %d\n", stats->vowels);
    printf("Consonants:  %d\n", stats->consonants);
    printf("Digits:      %d\n", stats->digits);
    printf("Spaces:      %d\n", stats->spaces);

    if (stats->words > 0) {
        float avg_word_length = (float)(stats->characters - stats->spaces) / stats->words;
        printf("Avg word length: %.2f characters\n", avg_word_length);
    }

    if (stats->sentences > 0) {
        float avg_words_per_sentence = (float)stats->words / stats->sentences;
        printf("Avg words per sentence: %.2f\n", avg_words_per_sentence);
    }
}

// Check if character is a vowel
int isVowel(char c) {
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
}

// Convert string to lowercase
void toLowerCase(char* str) {
    for (int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
}

// Create a sample file for testing
void createSampleFile(void) {
    FILE* file = fopen("sample.txt", "w");

    if (file == NULL) {
        printf("Error: Could not create sample file!\n");
        return;
    }

    fprintf(file, "The quick brown fox jumps over the lazy dog.\n");
    fprintf(file, "This is a sample text file for analysis.\n");
    fprintf(file, "It contains multiple lines and words.\n");
    fprintf(file, "The word 'the' appears multiple times in this text.\n");
    fprintf(file, "We can analyze the frequency of words, count characters, and more!\n");
    fprintf(file, "Programming in C is fun and educational.\n");
    fprintf(file, "This text analyzer demonstrates file I/O and string processing.\n");
    fprintf(file, "You can use this program to analyze any text file.\n");

    fclose(file);
}

/*
 * PROJECT FEATURES:
 * - Comprehensive text analysis
 * - Word frequency counting
 * - Multiple statistics tracking
 * - Sample file generation
 * - User-friendly interface
 *
 * CONCEPTS DEMONSTRATED:
 * - File I/O (fopen, fgets, fgetc, rewind)
 * - String processing (strcpy, strcmp, strlen)
 * - Character analysis (isalpha, isdigit, isspace, tolower)
 * - Structures for organizing data
 * - Arrays and loops
 * - Functions and modular design
 *
 * ALGORITHM COMPLEXITY:
 * - Basic stats: O(n) where n is file size
 * - Word frequency: O(n * m) where m is unique words
 * - Could be optimized with hash table
 *
 * POSSIBLE ENHANCEMENTS:
 * 1. Add spell checker functionality
 * 2. Generate word cloud data
 * 3. Identify longest/shortest word
 * 4. Count specific word occurrences
 * 5. Find palindromes in text
 * 6. Calculate reading difficulty score
 * 7. Generate summary statistics report
 * 8. Support multiple file formats
 * 9. Add sentence analysis
 * 10. Implement search and replace
 * 11. Count unique words
 * 12. Generate word histogram
 */
