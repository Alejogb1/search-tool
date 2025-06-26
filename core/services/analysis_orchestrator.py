# The main service that runs the pipeline

import sys
import os
import asyncio

"get current working directory"
sys.path.append(os.getcwd())
from integrations.nebulakw.search_engines import SearchEngineSuggestions as SES
from integrations.nebulakw.file_writter import FileWriter as FW
from integrations.nebulakw import config
from string import ascii_lowercase

from data_access import repository, models
from data_access.database import SessionLocal
def perform_research():
    try:
        with open('input-keywords.txt', encoding='utf-8') as seed_kws:
            input_kws = [kw.strip('\n').strip() for kw in seed_kws.readlines() if kw not in ('', ' ', '\n', None)]

        if len(input_kws) == 0:
            print(f'Please insert at least one keyword in input-keywords.txt to start, and try again.')
            sys.exit()

        kw_getters = {'Google': SES.get_google,
                    'Bing': SES.get_bing,
                    'Yahoo': SES.get_yahoo,
                    #'Yandex US': SES.get_yandex_us, RANDOM JSON FORMAT SOMETIMES
                    'Ask': SES.get_ask,
                    'DuckDuckGo': SES.get_duckduckgo,
                    'Ecosia': SES.get_ecosia,
                    'Brave': SES.get_brave
                    }

        for kw in input_kws:
            kws = tuple()
            for letter in ascii_lowercase:
                new_kw = f'{kw} {letter}'
                for se, getter in kw_getters.items():
                    try:
                        print(f'Exploring for {new_kw} in {se}')
                        kw_ses = SES.kw_results(getter, new_kw)
                        kws += kw_ses
                        print(f'{len(kw_ses)} keywords found in this step.\n')
                    except Exception as e:
                        print(f"Error exploring {new_kw} in {se}: {e}")
                        FW.write_log(f"Error exploring {new_kw} in {se}: {e}")
            FW.write_txt(kws)
        with open(config.output_file, encoding='utf-8') as out_f:
            print(f'File {config.output_file} has been created, '
                f'{len(out_f.readlines())} unique keywords found.')

    except Exception as e:
        print(f"Error running nebula kw: {e}")
        sys.exit(1)




class AnalysisOrchestrator:
    def __init__(self):
        self.db = SessionLocal()

    async def run_full_analysis(self, domain_url:str):
        db_domain = None,
        db = self.db


        
        #seed_keywords = generate(domain_url)
        
        perform_research()

        with open("output-keywords.txt", encoding='utf-8') as f:
            seed_keywords = [kw.strip('\n').strip() for kw in f.readlines() if kw not in ('', ' ', '\n', None)]


        try:
            # Create the logic to store seed keywords in the database, linked to the domain.

            repository.create_domain_and_keywords(db, domain_url, seed_keywords)

            # here it is missing job update to be completed
        
        except Exception as e:
            print(f"Error running analysis: {e}")
            sys.exit(1)
        finally:
            db.close()
    
async def main():
    # Create an instance of AnalysisOrchestrator
    orchestrator_instance = AnalysisOrchestrator()

    # Call the asynchronous method using await
    await orchestrator_instance.run_full_analysis(domain_url="lans.co")

# Run the main asynchronous function
if __name__ == "__main__":
    asyncio.run(main())
