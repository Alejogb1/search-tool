from sqlalchemy.orm import Session, joinedload
from . import models

def create_domain_if_not_exists(db: Session, domain_url: str) -> models.Domain:
    """
    Finds a Domain by its URL or creates a new one if it doesn't exist.
    Sets the status to PROCESSING.
    """
    print(f"Repository: Checking for domain '{domain_url}'.")
    db_domain = db.query(models.Domain).filter(models.Domain.url == domain_url).first()
    
    if db_domain:
        print(f"Repository: Domain ID {db_domain.id} already exists. Updating status.")
        db_domain.status = models.JobStatus.PROCESSING
    else:
        print(f"Repository: Domain not found. Creating new record.")
        db_domain = models.Domain(url=domain_url, status=models.JobStatus.PROCESSING)
        db.add(db_domain)
    
    db.commit()
    db.refresh(db_domain)
    print(f"Repository: Domain ID {db_domain.id} is ready for processing.")
    return db_domain

def create_keywords_for_domain(db: Session, domain_id: int, keywords_data: list[dict]):
    """
    Bulk-inserts keyword records from a list of dictionaries.
    This is designed to be highly efficient for large volumes of keywords.
    """
    if not keywords_data:
        print("Repository: No keyword data provided to insert.")
        return

    print(f"Repository: Preparing to bulk-insert {len(keywords_data)} keywords for domain ID {domain_id}.")
    
    # Prepare Keyword objects for bulk insertion from dictionaries
    keyword_objects = [
        models.Keyword(
            text=kw.get('text'),
            avg_monthly_searches=kw.get('avg_monthly_searches'),
            competition_level=kw.get('competition'), # Ensure field name matches model
            domain_id=domain_id
        ) for kw in keywords_data
    ]
    
    # Use bulk_save_objects for high performance.
    db.bulk_save_objects(keyword_objects)
    
    # Commit the transaction.
    db.commit()
    
    print(f"Repository: Successfully saved {len(keyword_objects)} keywords.")

def get_domain_with_keywords(db: Session, domain_id: int) -> models.Domain | None:
    """
    Retrieves a domain and all its associated keywords using an efficient join.
    """
    print(f"Repository: Retrieving domain ID {domain_id} with all its keywords.")
    return db.query(models.Domain).options(joinedload(models.Domain.keywords)).filter(models.Domain.id == domain_id).first()
