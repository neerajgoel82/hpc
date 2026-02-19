/*
 * File: 06_enums_unions.c
 * Topic: Enumerations and Unions
 *
 * Enums and unions are special data types in C. Enums provide named
 * integer constants, while unions allow storing different data types
 * in the same memory location.
 *
 * Key Concepts:
 * - Enumerated types (enum)
 * - Union types
 * - Memory layout differences
 * - When to use each
 * - Practical applications
 */

#include <stdio.h>
#include <string.h>

// ==================== ENUMERATIONS ====================

// Basic enum
enum Day {
    SUNDAY,     // 0
    MONDAY,     // 1
    TUESDAY,    // 2
    WEDNESDAY,  // 3
    THURSDAY,   // 4
    FRIDAY,     // 5
    SATURDAY    // 6
};

// Enum with custom values
enum Status {
    SUCCESS = 0,
    ERROR = -1,
    PENDING = 1,
    NOT_FOUND = 404
};

// Enum for colors (can be used as flags)
enum Color {
    RED = 1,
    GREEN = 2,
    BLUE = 4,
    YELLOW = 8
};

// Enum with typedef
typedef enum {
    NORTH,
    SOUTH,
    EAST,
    WEST
} Direction;

// ==================== UNIONS ====================

// Basic union - all members share same memory
union Data {
    int i;
    float f;
    char str[20];
};

// Union with typedef
typedef union {
    unsigned int full_value;
    struct {
        unsigned char byte1;
        unsigned char byte2;
        unsigned char byte3;
        unsigned char byte4;
    } bytes;
} IntBytes;

// Tagged union - union with a type identifier
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_STRING
} DataType;

typedef struct {
    DataType type;
    union {
        int i;
        float f;
        char str[50];
    } value;
} TaggedData;

// Practical example: IP Address representation
typedef union {
    unsigned int address;
    struct {
        unsigned char b1, b2, b3, b4;
    } bytes;
} IPAddress;

// Function prototypes
void printDay(enum Day d);
const char* statusToString(enum Status s);
void demonstrateUnion(void);
void demonstrateTaggedUnion(void);
void demonstrateIPAddress(void);

int main() {
    // ==================== ENUMERATIONS DEMO ====================

    printf("=== Enumerations (enum) ===\n\n");

    // Using enum
    enum Day today = WEDNESDAY;
    enum Day weekend = SATURDAY;

    printf("Today is day number %d\n", today);
    printf("Weekend starts on day %d\n", weekend);

    printDay(today);
    printDay(FRIDAY);

    // Enum in switch statement
    printf("\n=== Enum in Switch ===\n");
    switch (today) {
        case MONDAY:
        case TUESDAY:
        case WEDNESDAY:
        case THURSDAY:
        case FRIDAY:
            printf("It's a weekday\n");
            break;
        case SATURDAY:
        case SUNDAY:
            printf("It's the weekend!\n");
            break;
    }

    // Enum with custom values
    printf("\n=== Status Codes ===\n");
    enum Status operation_result = SUCCESS;
    printf("Operation status: %s\n", statusToString(operation_result));

    operation_result = ERROR;
    printf("Operation status: %s\n", statusToString(operation_result));

    operation_result = NOT_FOUND;
    printf("Operation status: %s (code: %d)\n",
           statusToString(operation_result), operation_result);

    // Using typedef enum
    printf("\n=== Direction Enum ===\n");
    Direction player_direction = NORTH;
    printf("Player is facing: ");
    switch (player_direction) {
        case NORTH: printf("North\n"); break;
        case SOUTH: printf("South\n"); break;
        case EAST:  printf("East\n"); break;
        case WEST:  printf("West\n"); break;
    }

    // Enum as flags (bitwise operations)
    printf("\n=== Enum as Flags ===\n");
    int colors = RED | BLUE;  // Combine flags
    printf("Color flags: %d\n", colors);

    if (colors & RED) {
        printf("Has RED\n");
    }
    if (colors & GREEN) {
        printf("Has GREEN\n");
    } else {
        printf("Does not have GREEN\n");
    }
    if (colors & BLUE) {
        printf("Has BLUE\n");
    }

    // ==================== UNIONS DEMO ====================

    printf("\n\n=== Unions ===\n\n");
    demonstrateUnion();

    printf("\n=== Memory Layout ===\n");
    union Data d;
    printf("Size of union Data: %zu bytes\n", sizeof(d));
    printf("Size of int: %zu bytes\n", sizeof(int));
    printf("Size of float: %zu bytes\n", sizeof(float));
    printf("Size of char[20]: %zu bytes\n", sizeof(char[20]));
    printf("Union takes size of largest member!\n");

    printf("\n");
    demonstrateTaggedUnion();

    printf("\n");
    demonstrateIPAddress();

    // ==================== COMPARISON ====================

    printf("\n=== Struct vs Union ===\n");

    struct PersonStruct {
        int age;
        float salary;
        char name[20];
    };

    union PersonUnion {
        int age;
        float salary;
        char name[20];
    };

    printf("Size of struct: %zu bytes (sum of all members)\n",
           sizeof(struct PersonStruct));
    printf("Size of union: %zu bytes (size of largest member)\n",
           sizeof(union PersonUnion));

    return 0;
}

// Helper function to print day name
void printDay(enum Day d) {
    const char *days[] = {
        "Sunday", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday"
    };
    printf("Day: %s\n", days[d]);
}

// Convert status to string
const char* statusToString(enum Status s) {
    switch (s) {
        case SUCCESS: return "Success";
        case ERROR: return "Error";
        case PENDING: return "Pending";
        case NOT_FOUND: return "Not Found";
        default: return "Unknown";
    }
}

// Demonstrate basic union usage
void demonstrateUnion(void) {
    printf("=== Basic Union Demo ===\n");

    union Data data;

    // Store integer
    data.i = 42;
    printf("data.i = %d\n", data.i);

    // Store float (overwrites integer!)
    data.f = 3.14f;
    printf("data.f = %.2f\n", data.f);
    printf("data.i = %d (garbage - was overwritten!)\n", data.i);

    // Store string (overwrites float!)
    strcpy(data.str, "Hello");
    printf("data.str = %s\n", data.str);
    printf("data.f = %.2f (garbage - was overwritten!)\n", data.f);
}

// Demonstrate tagged union (safe union usage)
void demonstrateTaggedUnion(void) {
    printf("=== Tagged Union (Safe Usage) ===\n");

    TaggedData data1, data2, data3;

    // Store integer
    data1.type = TYPE_INT;
    data1.value.i = 100;

    // Store float
    data2.type = TYPE_FLOAT;
    data2.value.f = 2.718f;

    // Store string
    data3.type = TYPE_STRING;
    strcpy(data3.value.str, "Tagged Union");

    // Safe retrieval based on type
    TaggedData items[] = {data1, data2, data3};

    for (int i = 0; i < 3; i++) {
        printf("Item %d: ", i + 1);

        switch (items[i].type) {
            case TYPE_INT:
                printf("Integer = %d\n", items[i].value.i);
                break;
            case TYPE_FLOAT:
                printf("Float = %.3f\n", items[i].value.f);
                break;
            case TYPE_STRING:
                printf("String = %s\n", items[i].value.str);
                break;
        }
    }
}

// Demonstrate IP address union
void demonstrateIPAddress(void) {
    printf("=== IP Address Union ===\n");

    IPAddress ip;

    // Set as integer
    ip.address = 0xC0A80101;  // 192.168.1.1 in hex

    printf("IP as integer: 0x%08X\n", ip.address);
    printf("IP as bytes: %d.%d.%d.%d\n",
           ip.bytes.b4, ip.bytes.b3, ip.bytes.b2, ip.bytes.b1);

    // Set individual bytes
    ip.bytes.b1 = 192;
    ip.bytes.b2 = 168;
    ip.bytes.b3 = 0;
    ip.bytes.b4 = 1;

    printf("IP set by bytes: %d.%d.%d.%d\n",
           ip.bytes.b1, ip.bytes.b2, ip.bytes.b3, ip.bytes.b4);
    printf("IP as integer: 0x%08X\n", ip.address);
}

/*
 * ENUMERATIONS (enum):
 *
 * Purpose:
 * - Create named integer constants
 * - Improve code readability
 * - Type-safe alternative to #define
 * - Self-documenting code
 *
 * Syntax:
 * enum Name { CONST1, CONST2, CONST3 };
 * enum Name { CONST1 = value1, CONST2 = value2 };
 *
 * Default values:
 * - Start at 0 and increment by 1
 * - Can specify custom values
 * - Subsequent values continue from last
 *
 * When to use:
 * - Related named constants
 * - State machines
 * - Status codes
 * - Flags and options
 * - Menu choices
 *
 * ==========================================
 *
 * UNIONS:
 *
 * Purpose:
 * - Store different types in same memory
 * - Memory efficiency
 * - Type punning (viewing same data differently)
 * - Low-level data manipulation
 *
 * Syntax:
 * union Name { type1 member1; type2 member2; };
 *
 * Key characteristics:
 * - All members share same memory location
 * - Size = size of largest member
 * - Only one member valid at a time
 * - No automatic type tracking
 *
 * When to use:
 * - Memory-constrained environments
 * - Type conversion and inspection
 * - Hardware register access
 * - Variant data types
 * - Protocol parsing
 *
 * ==========================================
 *
 * TAGGED UNIONS:
 *
 * Concept:
 * - Union + type tag (discriminator)
 * - Safe union usage pattern
 * - Know which member is valid
 *
 * Pattern:
 * struct TaggedUnion {
 *     enum Type type;
 *     union { ... } value;
 * };
 *
 * Benefits:
 * - Type safety
 * - Clear intent
 * - Prevent mistakes
 * - Common in variant types
 *
 * ==========================================
 *
 * ENUM vs #define:
 *
 * Use enum when:
 * - Related constants
 * - Need type checking
 * - Want grouping
 * - Better debugging
 *
 * Use #define when:
 * - General constants
 * - Macros with parameters
 * - Conditional compilation
 * - Non-integer values
 *
 * ==========================================
 *
 * COMMON PATTERNS:
 *
 * 1. State machines:
 *    enum State { IDLE, RUNNING, PAUSED, STOPPED };
 *
 * 2. Error codes:
 *    enum Error { OK = 0, ERR_MEMORY = -1, ... };
 *
 * 3. Bit flags:
 *    enum Flags { FLAG_A = 1, FLAG_B = 2, FLAG_C = 4 };
 *
 * 4. Menu options:
 *    enum Menu { ADD = 1, DELETE, MODIFY, QUIT };
 *
 * 5. Type variants:
 *    Tagged union for different data types
 *
 * 6. Hardware registers:
 *    Union to access bytes/words differently
 *
 * ==========================================
 *
 * EXERCISES:
 *
 * 1. Create enum for card suits and ranks (playing cards)
 * 2. Implement traffic light state machine with enum
 * 3. Build calculator with enum for operations
 * 4. Create union for RGB and integer color representation
 * 5. Implement variant type system with tagged unions
 * 6. Make file type enum (TEXT, BINARY, etc.)
 * 7. Create union for network packet parsing
 * 8. Build menu system using enums
 * 9. Implement bit flag system with enum
 * 10. Create configuration system with tagged unions
 */
