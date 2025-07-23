import csv
from integrations.google_ads_client import GoogleAdsClient

class KeywordEnricher:
    def __init__(self, customer_id, output_file='output-keywords.csv'):
        self.google_ads_client = GoogleAdsClient()
        self.customer_id = customer_id
        self.output_file = output_file
        self._initialize_csv()

    def _initialize_csv(self):
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'avg_monthly_searches', 'competition'])

    def expand_keywords(self, seed_keywords):
        if not seed_keywords:
            return []

        all_keyword_ideas = []
        batch_size = 20

        for i in range(0, len(seed_keywords), batch_size):
            batch = seed_keywords[i:i + batch_size]
            try:
                keyword_ideas = self.google_ads_client.generate_keyword_ideas(
                    customer_id=self.customer_id,
                    keyword_texts=batch
                )
                if keyword_ideas:
                    self._append_to_csv(keyword_ideas)
                    all_keyword_ideas.extend(keyword_ideas)
            except Exception as e:
                print(f"An error occurred while expanding keywords for batch {i//batch_size + 1}: {e}")
        
        return all_keyword_ideas

    def _append_to_csv(self, keyword_ideas):
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'avg_monthly_searches', 'competition'])
            for idea in keyword_ideas:
                writer.writerow(idea)

if __name__ == '__main__':
    # This is an example of how to use the KeywordEnricher
    # You should replace 'your-customer-id' with your actual Google Ads customer ID.
    # You can also load it from a config file or environment variable.
    customer_id = "YOUR CUSTOMER ID"  # IMPORTANT: Replace with a valid customer ID
    
    # Read seed keywords from input-keywords.txt
    try:
        with open('input-keywords.txt', 'r') as f:
            initial_keywords = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: input-keywords.txt not found. Please create it with a list of seed keywords.")
        initial_keywords = []

    # Initialize the enricher
    enricher = KeywordEnricher(customer_id, output_file='output-keywords.csv')

    # Expand the initial keywords
    expanded_keywords = enricher.expand_keywords(initial_keywords)

    # Print the results
    if expanded_keywords:
        print("Expanded Keywords:")
        for keyword in expanded_keywords:
            print(
                f"- Text: {keyword['text']}, "
                f"Avg. Monthly Searches: {keyword['avg_monthly_searches']}, "
                f"Competition: {keyword['competition']}"
            )
    else:
        print("No keywords were expanded. Check your customer ID and API credentials.")
