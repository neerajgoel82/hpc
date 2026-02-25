/*
 * File: 03_linked_list.c
 * Topic: Linked Lists
 *
 * A linked list is a dynamic data structure where each element (node)
 * contains data and a pointer to the next node.
 *
 * Key Concepts:
 * - Node structure
 * - Creating nodes
 * - Inserting at beginning, end, and middle
 * - Deleting nodes
 * - Traversing the list
 * - Searching
 */

#include <stdio.h>
#include <stdlib.h>

// Node structure
typedef struct Node {
    int data;
    struct Node* next;
} Node;

// Function prototypes
Node* createNode(int data);
void insertAtBeginning(Node** head, int data);
void insertAtEnd(Node** head, int data);
void insertAtPosition(Node** head, int data, int position);
void deleteNode(Node** head, int data);
void printList(Node* head);
int search(Node* head, int data);
int getLength(Node* head);
void freeList(Node** head);

int main() {
    Node* head = NULL;  // Empty list

    printf("=== Creating Linked List ===\n");

    // Insert at beginning
    printf("\nInserting at beginning: 30, 20, 10\n");
    insertAtBeginning(&head, 30);
    insertAtBeginning(&head, 20);
    insertAtBeginning(&head, 10);
    printList(head);

    // Insert at end
    printf("\nInserting at end: 40, 50\n");
    insertAtEnd(&head, 40);
    insertAtEnd(&head, 50);
    printList(head);

    // Insert at position
    printf("\nInserting 25 at position 2\n");
    insertAtPosition(&head, 25, 2);
    printList(head);

    // Get length
    printf("\nList length: %d\n", getLength(head));

    // Search for element
    printf("\n=== Searching ===\n");
    int search_value = 30;
    int position = search(head, search_value);
    if (position != -1) {
        printf("Found %d at position %d\n", search_value, position);
    } else {
        printf("%d not found\n", search_value);
    }

    search_value = 100;
    position = search(head, search_value);
    if (position != -1) {
        printf("Found %d at position %d\n", search_value, position);
    } else {
        printf("%d not found\n", search_value);
    }

    // Delete node
    printf("\n=== Deleting Nodes ===\n");
    printf("Deleting 25\n");
    deleteNode(&head, 25);
    printList(head);

    printf("Deleting 10 (first node)\n");
    deleteNode(&head, 10);
    printList(head);

    printf("Deleting 50 (last node)\n");
    deleteNode(&head, 50);
    printList(head);

    // Free memory
    printf("\n=== Cleanup ===\n");
    printf("Freeing all nodes...\n");
    freeList(&head);
    printf("List is now empty\n");
    printList(head);

    return 0;
}

// Create a new node
Node* createNode(int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

// Insert at the beginning
void insertAtBeginning(Node** head, int data) {
    Node* newNode = createNode(data);
    newNode->next = *head;
    *head = newNode;
}

// Insert at the end
void insertAtEnd(Node** head, int data) {
    Node* newNode = createNode(data);

    // If list is empty
    if (*head == NULL) {
        *head = newNode;
        return;
    }

    // Traverse to the last node
    Node* temp = *head;
    while (temp->next != NULL) {
        temp = temp->next;
    }

    temp->next = newNode;
}

// Insert at specific position (0-indexed)
void insertAtPosition(Node** head, int data, int position) {
    if (position == 0) {
        insertAtBeginning(head, data);
        return;
    }

    Node* newNode = createNode(data);
    Node* temp = *head;

    // Traverse to position-1
    for (int i = 0; i < position - 1 && temp != NULL; i++) {
        temp = temp->next;
    }

    if (temp == NULL) {
        printf("Position out of range\n");
        free(newNode);
        return;
    }

    newNode->next = temp->next;
    temp->next = newNode;
}

// Delete first occurrence of a node with given data
void deleteNode(Node** head, int data) {
    if (*head == NULL) {
        printf("List is empty\n");
        return;
    }

    Node* temp = *head;

    // If head node needs to be deleted
    if (temp->data == data) {
        *head = temp->next;
        free(temp);
        return;
    }

    // Search for the node to delete
    Node* prev = NULL;
    while (temp != NULL && temp->data != data) {
        prev = temp;
        temp = temp->next;
    }

    // Node not found
    if (temp == NULL) {
        printf("Node with data %d not found\n", data);
        return;
    }

    // Unlink and free the node
    prev->next = temp->next;
    free(temp);
}

// Print the list
void printList(Node* head) {
    if (head == NULL) {
        printf("List is empty\n");
        return;
    }

    Node* temp = head;
    printf("List: ");
    while (temp != NULL) {
        printf("%d", temp->data);
        if (temp->next != NULL) {
            printf(" -> ");
        }
        temp = temp->next;
    }
    printf(" -> NULL\n");
}

// Search for a value and return position (-1 if not found)
int search(Node* head, int data) {
    Node* temp = head;
    int position = 0;

    while (temp != NULL) {
        if (temp->data == data) {
            return position;
        }
        temp = temp->next;
        position++;
    }

    return -1;  // Not found
}

// Get length of list
int getLength(Node* head) {
    int count = 0;
    Node* temp = head;

    while (temp != NULL) {
        count++;
        temp = temp->next;
    }

    return count;
}

// Free all nodes
void freeList(Node** head) {
    Node* current = *head;
    Node* next;

    while (current != NULL) {
        next = current->next;
        free(current);
        current = next;
    }

    *head = NULL;
}

/*
 * LINKED LIST ADVANTAGES:
 * - Dynamic size (grows and shrinks at runtime)
 * - Efficient insertion/deletion (O(1) if position known)
 * - No memory waste
 *
 * LINKED LIST DISADVANTAGES:
 * - No random access (must traverse from head)
 * - Extra memory for pointers
 * - Not cache-friendly
 *
 * TIME COMPLEXITIES:
 * - Insert at beginning: O(1)
 * - Insert at end: O(n) (or O(1) with tail pointer)
 * - Delete: O(n) (need to find the node)
 * - Search: O(n)
 * - Access by index: O(n)
 *
 * MEMORY MANAGEMENT:
 * - Always free() dynamically allocated nodes
 * - Be careful with pointers when deleting
 * - Check for NULL before dereferencing
 *
 * TYPES OF LINKED LISTS:
 * 1. Singly Linked List (this file)
 * 2. Doubly Linked List (prev and next pointers)
 * 3. Circular Linked List (last node points to first)
 *
 * EXERCISES:
 * 1. Reverse a linked list
 * 2. Find the middle element
 * 3. Detect if list has a cycle
 * 4. Merge two sorted linked lists
 * 5. Remove duplicates from sorted list
 * 6. Find nth node from the end
 * 7. Implement a doubly linked list
 * 8. Implement a circular linked list
 * 9. Sort a linked list
 * 10. Check if list is palindrome
 */
