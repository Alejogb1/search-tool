import json
import subprocess
from datetime import datetime
from pathlib import Path
import re

def render_template(template_path, variables):
    """Render template with variables"""
    with open(template_path, 'r') as file:
        content = file.read()
    
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        content = content.replace(placeholder, str(value))
    
    return content

def generate_pdf(md_path, pdf_path):
    """Convert Markdown to PDF using Pandoc"""
    try:
        subprocess.run(['pandoc', '-s', str(md_path), '-o', str(pdf_path)], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Pandoc not found. Please install Pandoc to generate PDF files.")
        return False

def main():
    # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    # Create output directory
    company_slug = re.sub(r'\W+', '', config['company_name']).lower()
    output_dir = Path('reports') / company_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate email
    email_content = render_template('email_template.md', config)
    email_path = output_dir / f"email_to_{config['contact_first_name'].lower()}.md"
    with open(email_path, 'w') as email_file:
        email_file.write(email_content)
    
    # Generate report
    report_content = render_template('report_template.md', config)
    report_md_path = output_dir / f"{company_slug}_report.md"
    with open(report_md_path, 'w') as report_file:
        report_file.write(report_content)
    
    # Convert to PDF
    report_pdf_path = output_dir / f"{company_slug}_report.pdf"
    if generate_pdf(report_md_path, report_pdf_path):
        print(f"PDF report generated at: {report_pdf_path}")
    
    print(f"Email template generated at: {email_path}")
    print(f"Report Markdown generated at: {report_md_path}")

if __name__ == "__main__":
    main()
