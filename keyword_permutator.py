import itertools

def generate_permutations(keyword):
    """
    Generates 2-word combinations from keywords with 3+ words
    For example, 'a b c' -> ['a b', 'a c', 'b c']
    """
    words = keyword.split()
    if len(words) < 3:
        return []

    all_combinations = []
    # Generate only 2-word combinations
    for indices in itertools.combinations(range(len(words)), 2):
        # Create the new keyword string from the words at the selected indices
        new_keyword = ' '.join(words[j] for j in indices)
        all_combinations.append(new_keyword)
            
    return all_combinations

def process_keyword_file(input_file, output_file):
    """
    Reads keywords from an input file, generates permutations for each,
    and writes the unique set of original keywords and their permutations
    to an output file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            seed_keywords = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    all_keywords = set(seed_keywords)
    
    print(f"Read {len(seed_keywords)} unique seed keywords from '{input_file}'.")

    for keyword in seed_keywords:
        if len(keyword.split()) >= 3: # Only process keywords with 3+ words
            permutations = generate_permutations(keyword)
            all_keywords.update(permutations)

    print(f"Generated a total of {len(all_keywords)} unique keywords (including originals and permutations).")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for keyword in sorted(list(all_keywords)):
                f.write(keyword + '\n')
        print(f"Successfully wrote all keywords to '{output_file}'.")
    except IOError as e:
        print(f"Error writing to output file '{output_file}': {e}")

if __name__ == "__main__":
    # We need to define which input file to use.
    # For now, let's assume a default name 'input-keywords.txt'
    # and output to 'permutated-keywords.txt'.
    # This can be changed to use command-line arguments if needed.
    input_filename = "input-keywords.txt"
    output_filename = "permutated-keywords.txt"
    process_keyword_file(input_filename, output_filename)
