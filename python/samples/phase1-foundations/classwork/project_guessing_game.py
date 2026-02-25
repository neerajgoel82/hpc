"""
Week 2 Project: Number Guessing Game

Create an interactive number guessing game that:
1. Generates a random number between 1 and 100
2. Asks the player to guess the number
3. Provides hints (too high/too low)
4. Counts the number of attempts
5. Allows multiple rounds
6. Tracks high score (fewest attempts)

Requirements:
- Use while or for loops
- Use if/elif/else for logic
- Format output nicely
- Handle invalid input
"""

import random

# TODO: Print game title and instructions


# TODO: Initialize high score (start with a large number)


# TODO: Create main game loop (play again feature)
# Use while True with break, or ask yes/no


    # TODO: Generate random number between 1 and 100
    # hint: random.randint(1, 100)


    # TODO: Initialize attempt counter


    # TODO: Create guessing loop
    # Keep looping until guess is correct


        # TODO: Get guess from player
        # Handle non-numeric input


        # TODO: Increment attempt counter


        # TODO: Check if guess is correct, too high, or too low
        # Provide appropriate feedback


        # TODO: When correct, show number of attempts
        # Update high score if this is the best


    # TODO: Ask if player wants to play again


# TODO: Print farewell message and final high score


# BONUS FEATURES TO ADD:
# 1. Difficulty levels (Easy: 1-50, Medium: 1-100, Hard: 1-500)
# 2. Limited number of attempts (like 7 tries)
# 3. Scoring system (fewer attempts = higher score)
# 4. Hints after multiple wrong guesses
# 5. Player name and statistics tracking
# 6. Sound effects (using print statements like "BOOM!" "Ding!")
# 7. ASCII art for wins/losses


# Example Game Flow:
"""
================================
    NUMBER GUESSING GAME
================================
I'm thinking of a number between 1 and 100.
Can you guess it?

Attempt #1
Enter your guess: 50
Too low! Try again.

Attempt #2
Enter your guess: 75
Too high! Try again.

Attempt #3
Enter your guess: 63
Too low! Try again.

Attempt #4
Enter your guess: 69
Too high! Try again.

Attempt #5
Enter your guess: 66
🎉 Congratulations! You guessed it in 5 attempts!

Play again? (yes/no): no
Thanks for playing! Your best score: 5 attempts
"""

print("\n--- Starter Template ---")
print("Use this structure to build your game:\n")
print("""
# 1. Print title
# 2. Set high_score = float('inf')
# 3. Start game loop (play_again = True)
#    4. Generate secret number
#    5. Set attempts = 0
#    6. Start guessing loop (while guess != secret)
#       7. Get input
#       8. Increment attempts
#       9. Check and give feedback
#   10. Congratulate player
#   11. Update high score if better
#   12. Ask to play again
# 13. Show final statistics
""")
