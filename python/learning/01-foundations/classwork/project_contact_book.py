"""
Week 3 Project: Contact Book

Create a contact management system that:
1. Stores contacts (name, phone, email, address)
2. Add new contacts
3. Search for contacts by name
4. Display all contacts
5. Update contact information
6. Delete contacts
7. Save/load from file (optional)

This project uses dictionaries, lists, and all Phase 1 concepts.
"""

# Contact book structure: list of dictionaries
# [
#     {"name": "Alice", "phone": "123-456-7890", "email": "alice@email.com", "address": "123 Main St"},
#     {"name": "Bob", "phone": "234-567-8901", "email": "bob@email.com", "address": "456 Oak Ave"}
# ]

# TODO: Initialize empty contacts list
contacts = []


def display_menu():
    """Display the main menu"""
    print("\n" + "=" * 40)
    print("       CONTACT BOOK")
    print("=" * 40)
    print("1. Add Contact")
    print("2. View All Contacts")
    print("3. Search Contact")
    print("4. Update Contact")
    print("5. Delete Contact")
    print("6. Exit")
    print("=" * 40)


def add_contact():
    """Add a new contact"""
    # TODO: Get contact information from user
    # TODO: Create contact dictionary
    # TODO: Add to contacts list
    # TODO: Print success message
    pass


def view_all_contacts():
    """Display all contacts"""
    # TODO: Check if contacts list is empty
    # TODO: Loop through contacts and display each one
    # TODO: Format nicely with borders
    pass


def search_contact():
    """Search for a contact by name"""
    # TODO: Get search term from user
    # TODO: Search through contacts (case-insensitive)
    # TODO: Display matching contacts
    # TODO: Handle "not found" case
    pass


def update_contact():
    """Update an existing contact"""
    # TODO: Get contact name to update
    # TODO: Find contact in list
    # TODO: Display current information
    # TODO: Ask what to update (menu of options)
    # TODO: Update the information
    # TODO: Print success message
    pass


def delete_contact():
    """Delete a contact"""
    # TODO: Get contact name to delete
    # TODO: Find and remove contact
    # TODO: Print success/failure message
    pass


# TODO: Main program loop
# Keep showing menu until user chooses to exit

def main():
    """Main program function"""
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-6): ")

        if choice == "1":
            add_contact()
        elif choice == "2":
            view_all_contacts()
        elif choice == "3":
            search_contact()
        elif choice == "4":
            update_contact()
        elif choice == "5":
            delete_contact()
        elif choice == "6":
            print("\nThank you for using Contact Book!")
            break
        else:
            print("\nInvalid choice! Please try again.")


# TODO: Run the program
# main()


# BONUS FEATURES TO ADD:
# 1. Data validation (phone format, email format)
# 2. Sort contacts alphabetically
# 3. Export contacts to CSV file
# 4. Import contacts from CSV file
# 5. Add categories/groups
# 6. Add birthday field and birthday reminders
# 7. Favorite contacts
# 8. Search by phone number or email
# 9. Duplicate detection
# 10. Backup/restore functionality


# Example expected output:
"""
========================================
       CONTACT BOOK
========================================
1. Add Contact
2. View All Contacts
3. Search Contact
4. Update Contact
5. Delete Contact
6. Exit
========================================

Enter your choice (1-6): 1

--- Add New Contact ---
Name: Alice Smith
Phone: 123-456-7890
Email: alice@email.com
Address: 123 Main St

✓ Contact added successfully!

========================================
       CONTACT BOOK
========================================
1. Add Contact
2. View All Contacts
3. Search Contact
4. Update Contact
5. Delete Contact
6. Exit
========================================

Enter your choice (1-6): 2

--- All Contacts (1) ---

Contact #1
Name:    Alice Smith
Phone:   123-456-7890
Email:   alice@email.com
Address: 123 Main St
"""

print("\nStarter template provided. Implement each function!")
print("Start with add_contact() and view_all_contacts() functions.")
