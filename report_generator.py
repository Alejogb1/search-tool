import google.generativeai as genai
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_report():
    """
    Generates a search behavior analysis report using a prompt template,
    cluster data, and the Google Gemini API.
    """
    # 1. Configure API Key
    api_key = os.getenv("GOOGLE_API_KEYS")
    if not api_key:
        print("ERROR: GOOGLE_API_KEYS not found in .env file.")
        print("Please add at least one key.")
        return

    # In case of multiple keys, use the first one
    genai.configure(api_key=api_key.split(',')[0])

    # 2. Define file paths
    prompt_template_path = 'forrester_style_report_prompt.txt'
    clusters_csv_path = 'clusters_with_relevance.csv'
    output_report_path = 'customer_search_behavior_report.md'

    # 3. Read the prompt template and cluster data
    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Read the CSV to be included in the context for the LLM
        with open(clusters_csv_path, 'r', encoding='utf-8') as f:
            clusters_csv_content = f.read()

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return

    # 4. Define parameters to fill in the prompt
    # These are specific to the planroadmap.com analysis
    domain_name = "planroadmap.com"
    product_description = "An AI-powered toolbox that helps professionals with ADHD overcome task paralysis and make consistent progress."
    target_audience = "Professionals with ADHD in complex, unstructured work environments."
    industry = "Productivity Software & Neurodiversity Support"
    
    # 5. Fill the placeholders in the prompt template
    final_prompt = prompt_template.format(
        DOMAIN_NAME=domain_name,
        PRODUCT_DESCRIPTION=product_description,
        TARGET_AUDIENCE=target_audience,
        CLUSTERS_CSV_PATH=f"The following CSV data:\n\n{clusters_csv_content}",
        INDUSTRY=industry
    )

    print("Successfully created the final prompt. Sending to the LLM...")
    
    # 6. Generate content using the Gemini API
    try:
        # Using a more recent and valid model name
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(final_prompt)

        if response.text:
            report_content = response.text
            print("Successfully received response from the LLM.")
        else:
            print("Error: Received an empty response from the LLM.")
            return

    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return

    # 7. Save the generated report to a Markdown file
    try:
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Successfully saved the report to {output_report_path}")
    except IOError as e:
        print(f"Error writing the report to file: {e}")

if __name__ == "__main__":
    generate_report()
