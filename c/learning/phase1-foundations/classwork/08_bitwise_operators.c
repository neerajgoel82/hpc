/*
 * File: 08_bitwise_operators.c
 * Topic: Bitwise Operators in C
 *
 * Bitwise operators work on individual bits of data. They're essential
 * for low-level programming, embedded systems, and performance optimization.
 *
 * Key Concepts:
 * - Bitwise AND (&)
 * - Bitwise OR (|)
 * - Bitwise XOR (^)
 * - Bitwise NOT (~)
 * - Left shift (<<)
 * - Right shift (>>)
 * - Practical applications
 */

#include <stdio.h>

// Helper function to print binary representation
void printBinary(unsigned int n) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (n >> i) & 1);
        if (i % 4 == 0) printf(" ");
    }
}

int main() {
    // Bitwise AND (&)
    printf("=== Bitwise AND (&) ===\n");
    unsigned int a = 12;  // 1100 in binary
    unsigned int b = 10;  // 1010 in binary

    printf("a = %d (", a);
    printBinary(a);
    printf(")\n");

    printf("b = %d (", b);
    printBinary(b);
    printf(")\n");

    printf("a & b = %d (", a & b);
    printBinary(a & b);
    printf(")\n");
    printf("Rule: 1 & 1 = 1, otherwise 0\n");

    // Bitwise OR (|)
    printf("\n=== Bitwise OR (|) ===\n");
    printf("a | b = %d (", a | b);
    printBinary(a | b);
    printf(")\n");
    printf("Rule: 0 | 0 = 0, otherwise 1\n");

    // Bitwise XOR (^)
    printf("\n=== Bitwise XOR (^) ===\n");
    printf("a ^ b = %d (", a ^ b);
    printBinary(a ^ b);
    printf(")\n");
    printf("Rule: same = 0, different = 1\n");

    // Bitwise NOT (~)
    printf("\n=== Bitwise NOT (~) ===\n");
    unsigned char c = 5;  // 0000 0101
    printf("c = %d (", c);
    printBinary(c);
    printf(")\n");

    printf("~c = %d (", (unsigned char)~c);
    printBinary((unsigned char)~c);
    printf(")\n");
    printf("Rule: flips all bits (0->1, 1->0)\n");

    // Left Shift (<<)
    printf("\n=== Left Shift (<<) ===\n");
    unsigned int x = 5;  // 0101
    printf("x = %d (", x);
    printBinary(x);
    printf(")\n");

    printf("x << 1 = %d (", x << 1);
    printBinary(x << 1);
    printf(") (multiply by 2)\n");

    printf("x << 2 = %d (", x << 2);
    printBinary(x << 2);
    printf(") (multiply by 4)\n");

    printf("Note: Left shift by n = multiply by 2^n\n");

    // Right Shift (>>)
    printf("\n=== Right Shift (>>) ===\n");
    unsigned int y = 20;  // 0001 0100
    printf("y = %d (", y);
    printBinary(y);
    printf(")\n");

    printf("y >> 1 = %d (", y >> 1);
    printBinary(y >> 1);
    printf(") (divide by 2)\n");

    printf("y >> 2 = %d (", y >> 2);
    printBinary(y >> 2);
    printf(") (divide by 4)\n");

    printf("Note: Right shift by n = divide by 2^n\n");

    // Practical Application 1: Check if number is even/odd
    printf("\n=== Application 1: Check Even/Odd ===\n");
    int num = 7;
    if (num & 1) {
        printf("%d is odd\n", num);
    } else {
        printf("%d is even\n", num);
    }
    printf("Explanation: Last bit is 1 for odd, 0 for even\n");

    // Practical Application 2: Swap without temp variable
    printf("\n=== Application 2: Swap Using XOR ===\n");
    int p = 5, q = 10;
    printf("Before: p = %d, q = %d\n", p, q);

    p = p ^ q;
    q = p ^ q;
    p = p ^ q;

    printf("After: p = %d, q = %d\n", p, q);
    printf("Works because: a ^ a = 0, a ^ 0 = a\n");

    // Practical Application 3: Set, Clear, Toggle bits
    printf("\n=== Application 3: Bit Manipulation ===\n");
    unsigned char flags = 0;  // 0000 0000

    printf("Initial flags: ");
    printBinary(flags);
    printf("\n");

    // Set bit 2 (make it 1)
    flags = flags | (1 << 2);
    printf("After setting bit 2: ");
    printBinary(flags);
    printf("\n");

    // Set bit 5
    flags = flags | (1 << 5);
    printf("After setting bit 5: ");
    printBinary(flags);
    printf("\n");

    // Clear bit 2 (make it 0)
    flags = flags & ~(1 << 2);
    printf("After clearing bit 2: ");
    printBinary(flags);
    printf("\n");

    // Toggle bit 5 (flip it)
    flags = flags ^ (1 << 5);
    printf("After toggling bit 5: ");
    printBinary(flags);
    printf("\n");

    // Practical Application 4: Check if bit is set
    printf("\n=== Application 4: Check Bit Status ===\n");
    unsigned char value = 40;  // 0010 1000
    printf("Value: ");
    printBinary(value);
    printf("\n");

    int bit_position = 3;
    if (value & (1 << bit_position)) {
        printf("Bit %d is SET (1)\n", bit_position);
    } else {
        printf("Bit %d is CLEAR (0)\n", bit_position);
    }

    bit_position = 5;
    if (value & (1 << bit_position)) {
        printf("Bit %d is SET (1)\n", bit_position);
    } else {
        printf("Bit %d is CLEAR (0)\n", bit_position);
    }

    // Practical Application 5: Count set bits
    printf("\n=== Application 5: Count Set Bits ===\n");
    unsigned int number = 29;  // 0001 1101
    printf("Number: %d (", number);
    printBinary(number);
    printf(")\n");

    int count = 0;
    unsigned int temp = number;
    while (temp) {
        count += temp & 1;
        temp >>= 1;
    }
    printf("Number of set bits: %d\n", count);

    // Practical Application 6: Power of 2 check
    printf("\n=== Application 6: Check Power of 2 ===\n");
    int nums[] = {8, 15, 16, 31, 32};

    for (int i = 0; i < 5; i++) {
        // Power of 2 has only one bit set
        // n & (n-1) == 0 for powers of 2
        if (nums[i] > 0 && (nums[i] & (nums[i] - 1)) == 0) {
            printf("%d is a power of 2\n", nums[i]);
        } else {
            printf("%d is NOT a power of 2\n", nums[i]);
        }
    }

    // Practical Application 7: Extract/Pack values
    printf("\n=== Application 7: RGB Color Packing ===\n");
    // Pack RGB into single integer (commonly used in graphics)
    unsigned char red = 255, green = 128, blue = 64;
    unsigned int color = (red << 16) | (green << 8) | blue;

    printf("Color components: R=%d, G=%d, B=%d\n", red, green, blue);
    printf("Packed color: 0x%06X\n", color);

    // Unpack
    unsigned char r_extracted = (color >> 16) & 0xFF;
    unsigned char g_extracted = (color >> 8) & 0xFF;
    unsigned char b_extracted = color & 0xFF;

    printf("Unpacked: R=%d, G=%d, B=%d\n", r_extracted, g_extracted, b_extracted);

    // Practical Application 8: Bit masks and flags
    printf("\n=== Application 8: File Permissions (like Unix) ===\n");
    #define READ    (1 << 2)  // 100 (4)
    #define WRITE   (1 << 1)  // 010 (2)
    #define EXECUTE (1 << 0)  // 001 (1)

    unsigned char permissions = 0;

    // Grant read and write
    permissions = READ | WRITE;
    printf("Permissions: ");
    printBinary(permissions);
    printf("\n");

    // Check permissions
    if (permissions & READ) printf("Can read\n");
    if (permissions & WRITE) printf("Can write\n");
    if (permissions & EXECUTE) printf("Can execute\n");
    else printf("Cannot execute\n");

    // Add execute permission
    permissions |= EXECUTE;
    printf("After adding execute: ");
    printBinary(permissions);
    printf("\n");

    // Revoke write permission
    permissions &= ~WRITE;
    printf("After removing write: ");
    printBinary(permissions);
    printf("\n");

    return 0;
}

/*
 * BITWISE OPERATORS SUMMARY:
 *
 * Operator | Name       | Description
 * ---------|------------|----------------------------------
 * &        | AND        | 1 if both bits are 1
 * |        | OR         | 1 if at least one bit is 1
 * ^        | XOR        | 1 if bits are different
 * ~        | NOT        | Inverts all bits
 * <<       | Left Shift | Shift bits left (multiply by 2^n)
 * >>       | Right Shift| Shift bits right (divide by 2^n)
 *
 * COMMON PATTERNS:
 *
 * Set bit n:       num |= (1 << n)
 * Clear bit n:     num &= ~(1 << n)
 * Toggle bit n:    num ^= (1 << n)
 * Check bit n:     num & (1 << n)
 * Check even:      num & 1 == 0
 * Check power of 2: num & (num - 1) == 0
 * Swap:            a ^= b; b ^= a; a ^= b;
 *
 * PRACTICAL USES:
 * - System programming and embedded systems
 * - Graphics programming (color packing)
 * - Cryptography and hashing
 * - Compression algorithms
 * - Network protocols
 * - Device drivers
 * - Performance optimization
 * - Flag management
 *
 * EXERCISES:
 * 1. Write a function to reverse the bits of a number
 * 2. Find the only non-repeating element in array where all others appear twice
 * 3. Count number of bits that need to be flipped to convert A to B
 * 4. Check if a number is a palindrome in binary
 * 5. Find position of rightmost set bit
 * 6. Implement multiplication using bitwise operators
 * 7. Create a bit vector for tracking presence of numbers
 * 8. Write functions for bit rotation (left and right)
 */
