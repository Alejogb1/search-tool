# data_access/repository.py
from sqlalchemy.orm import Session
from . import models

def create_domain_and_keywords(db: Session, domain_url: str, keywords_text: list[str]) -> models.Domain:
    """
    Creates a new Domain record and bulk-inserts its associated keywords in a single transaction.
    This is the primary function for storing initial analysis results.
    """
    print(f"Repository: Creating domain for '{domain_url}' with {len(keywords_text)} keywords.")
    
    # 1. Create the parent Domain object
    db_domain = models.Domain(url=domain_url, status=models.JobStatus.PROCESSING)
    db.add(db_domain)
    
    # 2. Flush the session. This assigns an ID to `db_domain` without committing the
    # transaction. We need this ID for the foreign key in the keywords.
    db.flush() 

    # 3. Prepare Keyword objects for bulk insertion
    keyword_objects = [
        models.Keyword(text=kw, domain_id=db_domain.id) for kw in keywords_text
    ]
    
    # 4. Use bulk_save_objects for high performance. This is much faster than adding one by one.
    db.bulk_save_objects(keyword_objects)
    
    # 5. Commit the entire transaction. If any part fails, everything is rolled back.
    db.commit()
    
    # 6. Refresh the object to get the final state from the DB (e.g., timestamps)
    db.refresh(db_domain)
    
    print(f"Repository: Successfully saved domain ID {db_domain.id}.")
    return db_domain    