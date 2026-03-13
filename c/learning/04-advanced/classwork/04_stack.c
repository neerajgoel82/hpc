/*
 * File: 04_stack.c
 * Topic: Stack Data Structure
 *
 * A stack is a Last-In-First-Out (LIFO) data structure.
 * Think of it like a stack of plates: you add and remove from the top.
 *
 * Key Concepts:
 * - Push (add to top)
 * - Pop (remove from top)
 * - Peek (view top without removing)
 * - isEmpty check
 * - Array-based implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAX_SIZE 100

// Stack structure (array-based)
typedef struct {
    int items[MAX_SIZE];
    int top;
} Stack;

// Function prototypes
void initStack(Stack* s);
bool isEmpty(Stack* s);
bool isFull(Stack* s);
void push(Stack* s, int value);
int pop(Stack* s);
int peek(Stack* s);
int size(Stack* s);
void printStack(Stack* s);

// Application functions
bool isBalanced(char* expression);
int evaluatePostfix(char* expression);
void decimalToBinary(int decimal);

int main() {
    Stack s;
    initStack(&s);

    printf("=== Basic Stack Operations ===\n");

    // Push elements
    printf("\nPushing: 10, 20, 30, 40, 50\n");
    push(&s, 10);
    push(&s, 20);
    push(&s, 30);
    push(&s, 40);
    push(&s, 50);
    printStack(&s);

    // Peek
    printf("\nTop element: %d\n", peek(&s));
    printf("Stack size: %d\n", size(&s));

    // Pop elements
    printf("\n=== Popping Elements ===\n");
    printf("Popped: %d\n", pop(&s));
    printf("Popped: %d\n", pop(&s));
    printStack(&s);

    printf("Top element now: %d\n", peek(&s));

    // Check if empty
    printf("\n=== Emptying Stack ===\n");
    while (!isEmpty(&s)) {
        printf("Popped: %d\n", pop(&s));
    }
    printf("Stack is %s\n", isEmpty(&s) ? "empty" : "not empty");

    // Application 1: Balanced Parentheses
    printf("\n=== Application: Balanced Parentheses ===\n");
    char* expressions[] = {
        "(())",
        "((()))",
        "(()())",
        "(()",
        "())(",
        "{[()]}",
        "{[(])}"
    };

    for (int i = 0; i < 7; i++) {
        printf("%s is %s\n", expressions[i],
               isBalanced(expressions[i]) ? "balanced" : "not balanced");
    }

    // Application 2: Decimal to Binary
    printf("\n=== Application: Decimal to Binary ===\n");
    int decimals[] = {10, 25, 42, 100};
    for (int i = 0; i < 4; i++) {
        printf("%d in binary: ", decimals[i]);
        decimalToBinary(decimals[i]);
        printf("\n");
    }

    // Application 3: Reverse a string
    printf("\n=== Application: Reverse String ===\n");
    char str[] = "Hello, World!";
    printf("Original: %s\n", str);

    Stack char_stack;
    initStack(&char_stack);

    // Push characters
    for (int i = 0; str[i] != '\0'; i++) {
        push(&char_stack, str[i]);
    }

    // Pop to reverse
    printf("Reversed: ");
    while (!isEmpty(&char_stack)) {
        printf("%c", (char)pop(&char_stack));
    }
    printf("\n");

    return 0;
}

// Initialize stack
void initStack(Stack* s) {
    s->top = -1;  // -1 indicates empty stack
}

// Check if stack is empty
bool isEmpty(Stack* s) {
    return s->top == -1;
}

// Check if stack is full
bool isFull(Stack* s) {
    return s->top == MAX_SIZE - 1;
}

// Push element onto stack
void push(Stack* s, int value) {
    if (isFull(s)) {
        printf("Stack overflow! Cannot push %d\n", value);
        return;
    }
    s->items[++(s->top)] = value;
}

// Pop element from stack
int pop(Stack* s) {
    if (isEmpty(s)) {
        printf("Stack underflow! Cannot pop from empty stack\n");
        return -1;
    }
    return s->items[(s->top)--];
}

// Peek at top element without removing
int peek(Stack* s) {
    if (isEmpty(s)) {
        printf("Stack is empty\n");
        return -1;
    }
    return s->items[s->top];
}

// Get stack size
int size(Stack* s) {
    return s->top + 1;
}

// Print stack contents
void printStack(Stack* s) {
    if (isEmpty(s)) {
        printf("Stack is empty\n");
        return;
    }

    printf("Stack (top to bottom): ");
    for (int i = s->top; i >= 0; i--) {
        printf("%d ", s->items[i]);
    }
    printf("\n");
}

// Check if parentheses are balanced
bool isBalanced(char* expression) {
    Stack s;
    initStack(&s);

    for (int i = 0; expression[i] != '\0'; i++) {
        char ch = expression[i];

        // Push opening brackets
        if (ch == '(' || ch == '[' || ch == '{') {
            push(&s, ch);
        }
        // Check closing brackets
        else if (ch == ')' || ch == ']' || ch == '}') {
            if (isEmpty(&s)) {
                return false;
            }

            char top = pop(&s);

            if ((ch == ')' && top != '(') ||
                (ch == ']' && top != '[') ||
                (ch == '}' && top != '{')) {
                return false;
            }
        }
    }

    return isEmpty(&s);  // Should be empty if balanced
}

// Convert decimal to binary using stack
void decimalToBinary(int decimal) {
    if (decimal == 0) {
        printf("0");
        return;
    }

    Stack s;
    initStack(&s);

    while (decimal > 0) {
        push(&s, decimal % 2);
        decimal /= 2;
    }

    while (!isEmpty(&s)) {
        printf("%d", pop(&s));
    }
}

/*
 * STACK OPERATIONS TIME COMPLEXITY:
 * - Push: O(1)
 * - Pop: O(1)
 * - Peek: O(1)
 * - isEmpty: O(1)
 * - isFull: O(1)
 *
 * STACK PROPERTIES:
 * - LIFO (Last In First Out)
 * - Fixed or dynamic size
 * - Only top element is accessible
 *
 * STACK APPLICATIONS:
 * 1. Function call management (call stack)
 * 2. Expression evaluation
 * 3. Backtracking algorithms
 * 4. Undo/Redo functionality
 * 5. Browser history (back button)
 * 6. Balanced parentheses checking
 * 7. Depth-first search
 * 8. Memory management
 *
 * ARRAY vs LINKED LIST IMPLEMENTATION:
 *
 * Array-based (this file):
 * + Fast access (no pointer overhead)
 * + Better cache performance
 * - Fixed size (can overflow)
 *
 * Linked List-based:
 * + Dynamic size
 * + No overflow (until system memory exhausted)
 * - Pointer overhead
 * - Slower (cache misses)
 *
 * COMMON MISTAKES:
 * - Not checking for overflow before push
 * - Not checking for underflow before pop
 * - Forgetting to initialize top to -1
 *
 * EXERCISES:
 * 1. Implement stack using linked list
 * 2. Evaluate infix expression using two stacks
 * 3. Convert infix to postfix notation
 * 4. Implement a min-stack (getMin in O(1))
 * 5. Sort a stack using another stack
 * 6. Implement stack with getMiddle operation
 * 7. Design a stack that supports push, pop, and getMax
 * 8. Reverse a stack using recursion
 * 9. Check for duplicate parentheses
 * 10. Implement browser back/forward using stacks
 */
