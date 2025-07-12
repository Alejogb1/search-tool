import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_access import repository
from data_access.database import SessionLocal
from core.services.keyword_enricher import KeywordEnricher
# Import other services that will be created
# from core.services.keyword_generator import KeywordGenerator
# from core.services.keyword_clusterer import KeywordClusterer

class AnalysisOrchestrator:
    def __init__(self, db_session):
        self.db = db_session
        # self.keyword_generator = KeywordGenerator()
        # self.keyword_enricher = KeywordEnricher(customer_id) # customer_id will need to be managed
        # self.keyword_clusterer = KeywordClusterer()

    async def run_full_analysis(self, domain_url: str, customer_id: str):
        """
        Orchestrates the full analysis pipeline:
        1. Generate seed keywords.
        2. Enrich keywords with metrics.
        3. Cluster keywords.
        4. Build knowledge graph (future step).
        """
        print(f"Starting full analysis for domain: {domain_url}")
        db_domain = None
        
        try:
            # Step 1: Generate Seed Keywords (Future Implementation)
            # For now, we'll read from a file as a placeholder.
            print("Step 1: Generating seed keywords...")
            with open('input-keywords.txt', 'r', encoding='utf-8') as f:
                seed_keywords = [line.strip() for line in f if line.strip()]
            
            if not seed_keywords:
                print("No seed keywords found in input-keywords.txt. Aborting analysis.")
                return

            # Step 2: Enrich Keywords (using the refactored service)
            print("Step 2: Enriching keywords with search metrics...")
            enricher = KeywordEnricher(customer_id=customer_id, output_file='output-keywords.csv')
            enriched_keywords = enricher.expand_keywords(seed_keywords)
            
            if not enriched_keywords:
                print("No keywords were enriched. Check API credentials and customer ID.")
                # Decide if to proceed with only seed keywords
            
            # Store the enriched keywords in the database
            # This logic will replace the old repository call
            print("Storing keywords and metrics in the database...")
            db_domain = repository.create_domain_if_not_exists(self.db, domain_url)
            repository.create_keywords_for_domain(self.db, db_domain.id, enriched_keywords)
            print(f"{len(enriched_keywords)} enriched keywords stored for domain {db_domain.id}.")

            # Step 3: Cluster Keywords (Future Implementation)
            print("Step 3: Clustering keywords (not yet implemented)...")
            # embeddings = self.keyword_clusterer.generate_embeddings(enriched_keywords)
            # clusters = self.keyword_clusterer.perform_clustering(embeddings)
            # repository.store_clusters(self.db, clusters)

            # Step 4: Build Knowledge Graph (Future Implementation)
            print("Step 4: Building knowledge graph (not yet implemented)...")

            print("Analysis pipeline completed successfully.")

        except Exception as e:
            print(f"An error occurred during the analysis pipeline: {e}")
            # Optionally, add more robust error handling and rollback logic
        finally:
            print("Closing database session.")
            self.db.close()

async def main():
    # This is an example of how to run the orchestrator
    # In a real application, this would be triggered by an API call.
    db_session = SessionLocal()
    customer_id = "ID-123131"  # This should be managed securely (e.g., from config)
    orchestrator = AnalysisOrchestrator(db_session)
    await orchestrator.run_full_analysis(domain_url="https://www.example.com/", customer_id=customer_id)

if __name__ == "__main__":
    asyncio.run(main())
