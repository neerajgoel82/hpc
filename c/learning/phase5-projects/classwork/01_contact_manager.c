/*
 * File: 01_contact_manager.c
 * Topic: Contact Management System
 *
 * A practical project that combines:
 * - Structures
 * - Dynamic memory
 * - File I/O
 * - Functions
 * - User interaction
 *
 * Features:
 * - Add new contact
 * - Display all contacts
 * - Search contact by name
 * - Delete contact
 * - Save to file
 * - Load from file
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NAME 50
#define MAX_PHONE 15
#define MAX_EMAIL 50
#define FILENAME "contacts.txt"

// Contact structure
typedef struct {
    char name[MAX_NAME];
    char phone[MAX_PHONE];
    char email[MAX_EMAIL];
} Contact;

// Contact list structure
typedef struct {
    Contact* contacts;
    int count;
    int capacity;
} ContactList;

// Function prototypes
void initContactList(ContactList* list);
void addContact(ContactList* list);
void displayAllContacts(ContactList* list);
void searchContact(ContactList* list);
void deleteContact(ContactList* list);
void saveToFile(ContactList* list);
void loadFromFile(ContactList* list);
void freeContactList(ContactList* list);
void displayMenu(void);
void clearInputBuffer(void);

int main() {
    ContactList list;
    initContactList(&list);

    // Load existing contacts from file
    loadFromFile(&list);

    int choice;

    printf("====================================\n");
    printf("  Contact Management System\n");
    printf("====================================\n\n");

    while (1) {
        displayMenu();
        printf("Enter your choice: ");

        if (scanf("%d", &choice) != 1) {
            printf("Invalid input!\n");
            clearInputBuffer();
            continue;
        }
        clearInputBuffer();

        switch (choice) {
            case 1:
                addContact(&list);
                break;
            case 2:
                displayAllContacts(&list);
                break;
            case 3:
                searchContact(&list);
                break;
            case 4:
                deleteContact(&list);
                break;
            case 5:
                saveToFile(&list);
                printf("Contacts saved successfully!\n");
                break;
            case 6:
                printf("\nExiting... Saving contacts...\n");
                saveToFile(&list);
                freeContactList(&list);
                printf("Goodbye!\n");
                return 0;
            default:
                printf("Invalid choice! Please try again.\n");
        }

        printf("\nPress Enter to continue...");
        getchar();
    }

    return 0;
}

// Initialize contact list
void initContactList(ContactList* list) {
    list->capacity = 10;
    list->count = 0;
    list->contacts = (Contact*)malloc(list->capacity * sizeof(Contact));

    if (list->contacts == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
}

// Add a new contact
void addContact(ContactList* list) {
    // Resize if needed
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        Contact* temp = (Contact*)realloc(list->contacts,
                                          list->capacity * sizeof(Contact));
        if (temp == NULL) {
            printf("Memory reallocation failed!\n");
            return;
        }
        list->contacts = temp;
    }

    Contact* c = &list->contacts[list->count];

    printf("\n--- Add New Contact ---\n");
    printf("Name: ");
    fgets(c->name, MAX_NAME, stdin);
    c->name[strcspn(c->name, "\n")] = '\0';

    printf("Phone: ");
    fgets(c->phone, MAX_PHONE, stdin);
    c->phone[strcspn(c->phone, "\n")] = '\0';

    printf("Email: ");
    fgets(c->email, MAX_EMAIL, stdin);
    c->email[strcspn(c->email, "\n")] = '\0';

    list->count++;
    printf("\nContact added successfully!\n");
}

// Display all contacts
void displayAllContacts(ContactList* list) {
    if (list->count == 0) {
        printf("\nNo contacts to display.\n");
        return;
    }

    printf("\n--- All Contacts ---\n");
    printf("%-5s %-20s %-15s %-30s\n", "No.", "Name", "Phone", "Email");
    printf("-------------------------------------------------------------------\n");

    for (int i = 0; i < list->count; i++) {
        printf("%-5d %-20s %-15s %-30s\n",
               i + 1,
               list->contacts[i].name,
               list->contacts[i].phone,
               list->contacts[i].email);
    }
    printf("-------------------------------------------------------------------\n");
    printf("Total contacts: %d\n", list->count);
}

// Search for a contact by name
void searchContact(ContactList* list) {
    if (list->count == 0) {
        printf("\nNo contacts to search.\n");
        return;
    }

    char search_name[MAX_NAME];
    printf("\nEnter name to search: ");
    fgets(search_name, MAX_NAME, stdin);
    search_name[strcspn(search_name, "\n")] = '\0';

    printf("\n--- Search Results ---\n");
    int found = 0;

    for (int i = 0; i < list->count; i++) {
        if (strstr(list->contacts[i].name, search_name) != NULL) {
            printf("Name:  %s\n", list->contacts[i].name);
            printf("Phone: %s\n", list->contacts[i].phone);
            printf("Email: %s\n", list->contacts[i].email);
            printf("----------------------\n");
            found = 1;
        }
    }

    if (!found) {
        printf("No contacts found matching '%s'\n", search_name);
    }
}

// Delete a contact
void deleteContact(ContactList* list) {
    if (list->count == 0) {
        printf("\nNo contacts to delete.\n");
        return;
    }

    displayAllContacts(list);

    int index;
    printf("\nEnter contact number to delete (1-%d): ", list->count);
    scanf("%d", &index);
    clearInputBuffer();

    if (index < 1 || index > list->count) {
        printf("Invalid contact number!\n");
        return;
    }

    index--;  // Convert to 0-based index

    // Shift contacts to remove the deleted one
    for (int i = index; i < list->count - 1; i++) {
        list->contacts[i] = list->contacts[i + 1];
    }

    list->count--;
    printf("Contact deleted successfully!\n");
}

// Save contacts to file
void saveToFile(ContactList* list) {
    FILE* file = fopen(FILENAME, "w");

    if (file == NULL) {
        printf("Error: Could not open file for writing!\n");
        return;
    }

    fprintf(file, "%d\n", list->count);

    for (int i = 0; i < list->count; i++) {
        fprintf(file, "%s\n%s\n%s\n",
                list->contacts[i].name,
                list->contacts[i].phone,
                list->contacts[i].email);
    }

    fclose(file);
}

// Load contacts from file
void loadFromFile(ContactList* list) {
    FILE* file = fopen(FILENAME, "r");

    if (file == NULL) {
        return;  // File doesn't exist yet, that's okay
    }

    int count;
    if (fscanf(file, "%d\n", &count) != 1) {
        fclose(file);
        return;
    }

    // Resize if needed
    if (count > list->capacity) {
        list->capacity = count;
        Contact* temp = (Contact*)realloc(list->contacts,
                                          list->capacity * sizeof(Contact));
        if (temp == NULL) {
            printf("Memory allocation failed!\n");
            fclose(file);
            return;
        }
        list->contacts = temp;
    }

    for (int i = 0; i < count; i++) {
        fgets(list->contacts[i].name, MAX_NAME, file);
        list->contacts[i].name[strcspn(list->contacts[i].name, "\n")] = '\0';

        fgets(list->contacts[i].phone, MAX_PHONE, file);
        list->contacts[i].phone[strcspn(list->contacts[i].phone, "\n")] = '\0';

        fgets(list->contacts[i].email, MAX_EMAIL, file);
        list->contacts[i].email[strcspn(list->contacts[i].email, "\n")] = '\0';
    }

    list->count = count;
    fclose(file);

    printf("Loaded %d contact(s) from file.\n\n", count);
}

// Free contact list memory
void freeContactList(ContactList* list) {
    free(list->contacts);
    list->contacts = NULL;
    list->count = 0;
    list->capacity = 0;
}

// Display menu
void displayMenu(void) {
    printf("\n====================================\n");
    printf("1. Add New Contact\n");
    printf("2. Display All Contacts\n");
    printf("3. Search Contact\n");
    printf("4. Delete Contact\n");
    printf("5. Save Contacts\n");
    printf("6. Exit\n");
    printf("====================================\n");
}

// Clear input buffer
void clearInputBuffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

/*
 * PROJECT FEATURES:
 * - Dynamic array with automatic resizing
 * - File persistence
 * - CRUD operations (Create, Read, Update, Delete)
 * - User-friendly menu interface
 * - Input validation
 *
 * CONCEPTS DEMONSTRATED:
 * - Structures and typedef
 * - Dynamic memory allocation (malloc, realloc)
 * - File I/O (fopen, fprintf, fscanf)
 * - String manipulation (fgets, strcspn, strstr)
 * - Functions and modular programming
 * - Menu-driven program design
 *
 * POSSIBLE ENHANCEMENTS:
 * 1. Edit existing contact
 * 2. Sort contacts by name
 * 3. Export to CSV format
 * 4. Import from CSV
 * 5. Add multiple phone numbers per contact
 * 6. Add address field
 * 7. Search by phone or email
 * 8. Add contact groups/categories
 * 9. Add birthday field with reminders
 * 10. Implement data encryption
 */
