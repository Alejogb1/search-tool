import os
import csv
import time
from openai import OpenAI
from pathlib import Path

# Configuration
INPUT_CSV = 'output-keywords.csv'
OUTPUT_CSV = 'analyzed-keywords.csv'
DATABRICKS_ENDPOINT = "https://dbc-3bcdf049-c6b7.cloud.databricks.com/serving-endpoints"
MODEL_NAME = "databricks-gpt-oss-120b"
BATCH_SIZE = 10000  # Number of keywords to process per request
DELAY_SECONDS = 2  # Delay between requests to avoid rate limiting

def analyze_keywords(keywords):
    """Analyze keywords using Databricks GPT model"""
    client = OpenAI(
        api_key='dapi20ea086ba260b12a15fbb9941a5c5b33',
        base_url=DATABRICKS_ENDPOINT
    )
    
    results = []
    for i in range(0, len(keywords), BATCH_SIZE):
        batch = keywords[i:i+BATCH_SIZE]
        keyword_list = "\n- ".join(batch)
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert market intelligence analyst. Analyze search keywords to identify: "
                               "1. Search intent (informational, commercial, navigational, transactional) "
                               "2. Probable industry/domain "
                               "3. Key entities mentioned "
                               "4. Potential commercial value (low, medium, high) "
                               "Return results in CSV format: keyword,intent,industry,entities,value"
                },
                {
                    "role": "user",
                    "content": f"Analyze these search terms:\n- {keyword_list}"
                }
            ]
        )
        
        # Parse response
        content = response.choices[0].message.content
        if content:
            for line in content.split('\n'):
                if line and ',' in line:
                    results.append(line.split(','))
        
        # Avoid rate limiting
        time.sleep(DELAY_SECONDS)
    
    return results

def main():
    # Read input CSV
    keywords = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                keywords.append(row[0])  # Assuming first column contains keywords
    
    # Analyze keywords
    analyzed_data = analyze_keywords(keywords)
    
    # Write output CSV
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Keyword', 'Intent', 'Industry', 'Entities', 'Value'])
        writer.writerows(analyzed_data)
    
    print(f"Analysis complete! Results saved to {OUTPUT_CSV}")
    print(f"Total keywords analyzed: {len(analyzed_data)}")

if __name__ == "__main__":
    main()
