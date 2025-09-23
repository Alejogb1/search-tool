import re
import os

def clean_keywords_file(input_path: str, output_path: str):
    """Clean numerical prefixes from a keywords CSV file"""
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        return False
        
    with open(input_path, 'r') as f:
        lines = f.readlines()
        
    # Remove leading numbers/dots/spaces using regex
    cleaned = [re.sub(r'^\d+\.?\s*', '', line).strip() + '\n' for line in lines]
    
    with open(output_path, 'w') as f:
        f.writelines(cleaned)
        
    return len(cleaned)

if __name__ == "__main__":
    input_file = "output-keywords-databricks.com.csv"
    output_file = "output-keywords-databricks-cleaned.com.csv"
    
    cleaned_count = clean_keywords_file(input_file, output_file)
    if cleaned_count > 0:
        print(f"Successfully cleaned {cleaned_count} keywords")
        print(f"Output saved to: {os.path.abspath(output_file)}")
