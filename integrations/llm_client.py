from google import genai
from google.genai import types
from web_scraper import Webscraper
import asyncio


""" old_prompt =
Objective: To identify a statistically significant set of seed keywords for a given domain URL that maximizes the probability of attracting users within a specified target demographic.
Procedure:
1.  Domain URL: https://agentazlon.com/
2.  Target Demographic: Define the target demographic using measurable variables (e.g., age range, income level, geographic location, interests quantified through online behavior).
3.  Content Analysis: Perform a quantitative content analysis of the domain, categorizing content into distinct themes (T1, T2, ..., Tn) and assigning a probability distribution P(Ti) representing the relative frequency of each theme.
4.  Keyword Generation: Generate an initial set of candidate keywords (K1, K2, ..., Km) through brainstorming and automated tools.
5.  Relevance Scoring: Assign a relevance score R(Ki, Ti) to each keyword Ki for each theme Ti, based on expert judgment or automated semantic analysis.
6.  Probability Estimation: Estimate the probability P(User | Ki) that a user within the target demographic will search for keyword Ki, using historical search data or survey data.
7.  Expected Value Calculation: Calculate the expected value E(Ki) of each keyword Ki as:
    E(Ki) = Σ \[P(Ti) \* R(Ki, Ti) \* P(User | Ki)] for all i
8.  Keyword Selection: Select the top N keywords with the highest expected values, where N is a predetermined sample size.
9.  Hypothesis Testing: Formulate a null hypothesis (H0) that the selected keywords do not significantly outperform a randomly selected set of keywords in attracting users from the target demographic.
10. A/B Testing: Conduct A/B testing by comparing the performance of the selected keywords against a control group of randomly selected keywords.
11. Statistical Analysis: Perform statistical analysis (e.g., t-tests, chi-squared tests) to determine whether the observed difference in performance is statistically significant, rejecting H0 if p < α, where α is the significance level.
Deliverables:
*   A list of N statistically validated seed keywords.
*   A report detailing the methodology, data sources, calculations, and statistical analysis performed.
*   A justification for the selection of each keyword based on the expected value and statistical significance.

"""

n_keywords = 200

system_prompt = f"""
You are a Keyword Research Specialist. Your job is to generate comprehensive keyword lists based on domain analysis without making assumptions about search volume or user behavior.

DOMAIN ANALYSIS REQUIREMENTS:

Analyze the provided domain's actual content, services, and offerings

Extract real terminology used on the website

Identify actual problems mentioned or implied

Note competitor names or solutions referenced

List specific features, technologies, or processes described

KEYWORD GENERATION RULES:

Generate minimum {n_keywords} unique keywords

Use actual terminology from the domain, not assumed "natural" phrases

Include variations of real terms found on the site

Add logical extensions of actual content themes

Avoid inventing search patterns you cannot verify

KEYWORD CATEGORIES TO COVER:

Direct service/product terms from the site

Problem statements actually mentioned

Technology terms and processes described

Industry terminology used in content

Competitor names if referenced

Feature-specific terms

Process-related keywords

Alternative phrasings of site content

Focus on generic/solution terms, some brand name (but for competitors)

Prioritize problem-solution keywords based on actual site content

Include educational terms only if site positions as educational resource

OUTPUT FORMAT:
Plain TXT no column headers, followed by keyword list. No explanations, analysis, or commentary.

QUALITY REQUIREMENTS:

Cannot use the own's brand name

All keywords must relate to actual domain content

No artificially constructed phrases

No repetitive keyword stuffing patterns

Each keyword serves distinct search intent

Mix of 1-5 word phrases

Focus on terminology that appears on or relates to the actual website

Generate keywords based on what the domain actually offers and discusses, not theoretical user search patterns.

"""


def generate(domain_url:str):
    client = genai.Client(api_key="API KEY")
    scraper_response = Webscraper(domain_url)
    response = client.models.generate_content( 
        model = 'gemini-2.5-flash-preview-05-20', 
        contents= system_prompt + scraper_response.text        
        )
    with open('input-keywords.txt', 'w') as f:
        f.write(response.text)

    return response