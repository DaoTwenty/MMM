#!/bin/bash

# Read input string
read -p "Enter a string: " input

# Count number of words (space-separated)
word_count=$(echo "$input" | wc -w)

echo "Number of words: $word_count"