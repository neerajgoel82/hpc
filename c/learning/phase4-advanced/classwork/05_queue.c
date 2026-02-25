/*
 * File: 05_queue.c
 * Topic: Queue Data Structure
 *
 * A queue is a First-In-First-Out (FIFO) data structure.
 * Think of it like a line at a store: first person in line is served first.
 *
 * Key Concepts:
 * - Enqueue (add to rear)
 * - Dequeue (remove from front)
 * - Front and rear pointers
 * - Circular queue
 * - Array-based and linked-list implementations
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_SIZE 100

// ==================== ARRAY-BASED QUEUE ====================

typedef struct {
    int items[MAX_SIZE];
    int front;
    int rear;
    int size;
} Queue;

// Function prototypes for array-based queue
void initQueue(Queue* q);
bool isEmpty(Queue* q);
bool isFull(Queue* q);
void enqueue(Queue* q, int value);
int dequeue(Queue* q);
int peek(Queue* q);
int queueSize(Queue* q);
void printQueue(Queue* q);

// ==================== LINKED LIST-BASED QUEUE ====================

typedef struct QueueNode {
    int data;
    struct QueueNode* next;
} QueueNode;

typedef struct {
    QueueNode* front;
    QueueNode* rear;
    int size;
} LinkedQueue;

// Function prototypes for linked queue
void initLinkedQueue(LinkedQueue* q);
bool isLinkedQueueEmpty(LinkedQueue* q);
void enqueueLinked(LinkedQueue* q, int value);
int dequeueLinked(LinkedQueue* q);
int peekLinked(LinkedQueue* q);
int linkedQueueSize(LinkedQueue* q);
void printLinkedQueue(LinkedQueue* q);
void freeLinkedQueue(LinkedQueue* q);

// ==================== MAIN ====================

int main() {
    // Array-based queue demo
    printf("=== Array-Based Circular Queue ===\n\n");

    Queue q;
    initQueue(&q);

    printf("Enqueueing: 10, 20, 30, 40, 50\n");
    enqueue(&q, 10);
    enqueue(&q, 20);
    enqueue(&q, 30);
    enqueue(&q, 40);
    enqueue(&q, 50);
    printQueue(&q);

    printf("\nFront element: %d\n", peek(&q));
    printf("Queue size: %d\n", queueSize(&q));

    printf("\n=== Dequeue Operations ===\n");
    printf("Dequeued: %d\n", dequeue(&q));
    printf("Dequeued: %d\n", dequeue(&q));
    printQueue(&q);

    printf("\nEnqueueing: 60, 70\n");
    enqueue(&q, 60);
    enqueue(&q, 70);
    printQueue(&q);

    printf("\n=== Emptying Queue ===\n");
    while (!isEmpty(&q)) {
        printf("Dequeued: %d\n", dequeue(&q));
    }
    printf("Queue is %s\n", isEmpty(&q) ? "empty" : "not empty");

    // Linked list-based queue demo
    printf("\n\n=== Linked List-Based Queue ===\n\n");

    LinkedQueue lq;
    initLinkedQueue(&lq);

    printf("Enqueueing: 5, 10, 15, 20, 25\n");
    enqueueLinked(&lq, 5);
    enqueueLinked(&lq, 10);
    enqueueLinked(&lq, 15);
    enqueueLinked(&lq, 20);
    enqueueLinked(&lq, 25);
    printLinkedQueue(&lq);

    printf("\nFront element: %d\n", peekLinked(&lq));
    printf("Queue size: %d\n", linkedQueueSize(&lq));

    printf("\nDequeued: %d\n", dequeueLinked(&lq));
    printf("Dequeued: %d\n", dequeueLinked(&lq));
    printLinkedQueue(&lq);

    // Application: Breadth-First Search simulation
    printf("\n\n=== Application: BFS Simulation ===\n");
    Queue bfs_queue;
    initQueue(&bfs_queue);

    printf("Simulating BFS traversal starting from node 1\n");
    enqueue(&bfs_queue, 1);

    int visited_count = 0;
    while (!isEmpty(&bfs_queue) && visited_count < 7) {
        int node = dequeue(&bfs_queue);
        printf("Visiting node: %d\n", node);
        visited_count++;

        // Simulate adding neighbors
        if (node * 2 <= 7) enqueue(&bfs_queue, node * 2);
        if (node * 2 + 1 <= 7) enqueue(&bfs_queue, node * 2 + 1);
    }

    // Application: Task scheduling
    printf("\n\n=== Application: Task Scheduler ===\n");
    Queue task_queue;
    initQueue(&task_queue);

    printf("Adding tasks to queue:\n");
    enqueue(&task_queue, 101);  // Task IDs
    enqueue(&task_queue, 102);
    enqueue(&task_queue, 103);
    enqueue(&task_queue, 104);

    printf("Processing tasks in order:\n");
    while (!isEmpty(&task_queue)) {
        int task_id = dequeue(&task_queue);
        printf("Processing task %d\n", task_id);
    }

    // Cleanup
    freeLinkedQueue(&lq);

    return 0;
}

// ==================== ARRAY-BASED QUEUE IMPLEMENTATION ====================

void initQueue(Queue* q) {
    q->front = 0;
    q->rear = -1;
    q->size = 0;
}

bool isEmpty(Queue* q) {
    return q->size == 0;
}

bool isFull(Queue* q) {
    return q->size == MAX_SIZE;
}

void enqueue(Queue* q, int value) {
    if (isFull(q)) {
        printf("Queue overflow! Cannot enqueue %d\n", value);
        return;
    }

    // Circular increment
    q->rear = (q->rear + 1) % MAX_SIZE;
    q->items[q->rear] = value;
    q->size++;
}

int dequeue(Queue* q) {
    if (isEmpty(q)) {
        printf("Queue underflow! Cannot dequeue from empty queue\n");
        return -1;
    }

    int value = q->items[q->front];
    q->front = (q->front + 1) % MAX_SIZE;
    q->size--;

    return value;
}

int peek(Queue* q) {
    if (isEmpty(q)) {
        printf("Queue is empty\n");
        return -1;
    }
    return q->items[q->front];
}

int queueSize(Queue* q) {
    return q->size;
}

void printQueue(Queue* q) {
    if (isEmpty(q)) {
        printf("Queue is empty\n");
        return;
    }

    printf("Queue (front to rear): ");
    int index = q->front;
    for (int i = 0; i < q->size; i++) {
        printf("%d ", q->items[index]);
        index = (index + 1) % MAX_SIZE;
    }
    printf("\n");
}

// ==================== LINKED QUEUE IMPLEMENTATION ====================

void initLinkedQueue(LinkedQueue* q) {
    q->front = NULL;
    q->rear = NULL;
    q->size = 0;
}

bool isLinkedQueueEmpty(LinkedQueue* q) {
    return q->front == NULL;
}

void enqueueLinked(LinkedQueue* q, int value) {
    QueueNode* newNode = (QueueNode*)malloc(sizeof(QueueNode));
    if (newNode == NULL) {
        printf("Memory allocation failed!\n");
        return;
    }

    newNode->data = value;
    newNode->next = NULL;

    if (isLinkedQueueEmpty(q)) {
        // First element
        q->front = newNode;
        q->rear = newNode;
    } else {
        // Add to rear
        q->rear->next = newNode;
        q->rear = newNode;
    }

    q->size++;
}

int dequeueLinked(LinkedQueue* q) {
    if (isLinkedQueueEmpty(q)) {
        printf("Queue underflow! Cannot dequeue from empty queue\n");
        return -1;
    }

    QueueNode* temp = q->front;
    int value = temp->data;

    q->front = q->front->next;

    // If queue becomes empty
    if (q->front == NULL) {
        q->rear = NULL;
    }

    free(temp);
    q->size--;

    return value;
}

int peekLinked(LinkedQueue* q) {
    if (isLinkedQueueEmpty(q)) {
        printf("Queue is empty\n");
        return -1;
    }
    return q->front->data;
}

int linkedQueueSize(LinkedQueue* q) {
    return q->size;
}

void printLinkedQueue(LinkedQueue* q) {
    if (isLinkedQueueEmpty(q)) {
        printf("Queue is empty\n");
        return;
    }

    printf("Queue (front to rear): ");
    QueueNode* current = q->front;
    while (current != NULL) {
        printf("%d ", current->data);
        current = current->next;
    }
    printf("\n");
}

void freeLinkedQueue(LinkedQueue* q) {
    while (!isLinkedQueueEmpty(q)) {
        dequeueLinked(q);
    }
}

/*
 * QUEUE OPERATIONS TIME COMPLEXITY:
 * - Enqueue: O(1)
 * - Dequeue: O(1)
 * - Peek: O(1)
 * - isEmpty: O(1)
 * - isFull: O(1) (array-based only)
 *
 * QUEUE PROPERTIES:
 * - FIFO (First In First Out)
 * - Operations at both ends (front and rear)
 * - Fair ordering
 * - No random access
 *
 * QUEUE APPLICATIONS:
 * 1. Task scheduling (CPU, printer, etc.)
 * 2. Breadth-First Search (BFS) in graphs
 * 3. Buffer management (keyboard, I/O)
 * 4. Request handling in servers
 * 5. Call center systems
 * 6. Print queue
 * 7. Message queues
 * 8. Cache implementation (LRU)
 * 9. Simulation systems
 * 10. Handling interrupts
 *
 * ARRAY-BASED vs LINKED-LIST QUEUE:
 *
 * Array-Based (Circular):
 * + Fast access, better cache performance
 * + No pointer overhead
 * + Predictable memory usage
 * - Fixed size (can overflow)
 * - Wasted space if not full
 *
 * Linked-List Based:
 * + Dynamic size (no overflow)
 * + Efficient memory use
 * + No size limit (until system memory full)
 * - Pointer overhead
 * - Slower (cache misses)
 * - More complex
 *
 * CIRCULAR QUEUE:
 * - Uses modulo arithmetic to wrap around
 * - Efficient use of array space
 * - front = (front + 1) % MAX_SIZE
 * - rear = (rear + 1) % MAX_SIZE
 * - Avoids shifting elements
 *
 * QUEUE VARIANTS:
 * 1. Simple Queue (this file)
 * 2. Circular Queue (array-based in this file)
 * 3. Priority Queue (elements have priority)
 * 4. Double-ended Queue (Deque) - insert/delete both ends
 * 5. Concurrent Queue (thread-safe)
 *
 * COMMON PATTERNS:
 *
 * 1. Producer-Consumer:
 *    Producer adds to queue, consumer removes
 *
 * 2. Request buffering:
 *    Store requests when service is busy
 *
 * 3. Level-order traversal:
 *    BFS uses queue for tree/graph traversal
 *
 * 4. Cache replacement:
 *    FIFO cache eviction policy
 *
 * QUEUE vs STACK:
 * - Queue: FIFO, fair, BFS
 * - Stack: LIFO, recent first, DFS
 *
 * EXERCISES:
 * 1. Implement priority queue
 * 2. Create double-ended queue (deque)
 * 3. Implement queue using two stacks
 * 4. Build a print spooler simulation
 * 5. Implement circular buffer for streaming
 * 6. Create a ticket booking system using queue
 * 7. Simulate bank queue with multiple servers
 * 8. Implement sliding window maximum using queue
 * 9. Build a call center routing system
 * 10. Create LRU cache using queue
 * 11. Implement BFS for maze solving
 * 12. Build task scheduler with time slicing
 */
