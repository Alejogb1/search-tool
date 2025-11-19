from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql import func
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

def create_seed_keywords_for_domain(db: Session, domain_id: int, seed_keywords: list[str]):
    """
    Inserts seed keyword records from a list of strings.
    Seed keywords don't have search volume data.
    Uses individual inserts to properly handle func.now() timestamps.
    """
    if not seed_keywords:
        print("Repository: No seed keywords provided to insert.")
        return

    print(f"Repository: Preparing to insert {len(seed_keywords)} seed keywords for domain ID {domain_id}.")

    # Individual inserts to handle func.now() properly with PostgreSQL
    for keyword_text in seed_keywords:
        kw = models.Keyword(
            text=keyword_text,
            keyword_type=models.KeywordType.SEED_ONLY,  # Mark as seed-only keyword
            seeded_at=func.now(),  # Set seeded timestamp
            domain_id=domain_id
        )
        db.add(kw)

    # Commit the transaction.
    db.commit()

    print(f"Repository: Successfully saved {len(seed_keywords)} seed keywords.")

def update_keywords_with_expanded_data(db: Session, domain_id: int, expanded_keywords: list[dict], batch_hash: str = None):
    """
    Updates existing seed keywords with expanded data from Google Ads API.
    Creates new keyword records for keywords that weren't in the seed list.
    Includes batch tracking to prevent re-processing and enable resume capability.
    """
    if not expanded_keywords:
        print("Repository: No expanded keyword data provided to update.")
        return 0

    # Generate a batch hash if not provided (for non-batch workflows)
    if batch_hash is None:
        import hashlib
        import time
        hash_input = f"{domain_id}_{len(expanded_keywords)}_{int(time.time())}"
        batch_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]
        print(f"Repository: Generated batch hash {batch_hash} for non-batch workflow.")

    print(f"Repository: Processing batch {batch_hash} with {len(expanded_keywords)} keywords for domain ID {domain_id}.")

    # Check if batch already processed
    existing_batch = db.query(models.KeywordBatch).filter(
        models.KeywordBatch.domain_id == domain_id,
        models.KeywordBatch.batch_hash == batch_hash,
        models.KeywordBatch.processed == True
    ).first()

    if existing_batch:
        print(f"Repository: Batch {batch_hash} already processed. Skipping.")
        return 0

    # Get ONLY seed keywords (not previously expanded ones) for comparison
    seed_keywords = db.query(models.Keyword.text).filter(
        models.Keyword.domain_id == domain_id,
        models.Keyword.keyword_type == models.KeywordType.SEED_ONLY
    ).all()
    seed_text_set = {kw[0] for kw in seed_keywords}  # Extract text from tuple

    # Filter to only NEW keywords not in seed set
    new_keywords = [kw for kw in expanded_keywords if kw.get('text') not in seed_text_set]

    if not new_keywords:
        print(f"Repository: No new keywords to save for batch {batch_hash}. All keywords already exist.")
        # Mark batch as processed even if no new keywords
        mark_batch_processed(db, domain_id, batch_hash, 0)
        return 0

    # Bulk insert new keywords
    keyword_objects = []
    for kw_data in new_keywords:
        keyword_objects.append(models.Keyword(
            text=kw_data.get('text'),
            avg_monthly_searches=kw_data.get('avg_monthly_searches'),
            competition_level=kw_data.get('competition'),
            keyword_type=models.KeywordType.EXPANDED_ONLY,
            batch_info=batch_hash,
            domain_id=domain_id
        ))

    # Bulk save in batches to avoid memory issues
    batch_size = 1000
    saved_count = 0
    for i in range(0, len(keyword_objects), batch_size):
        batch = keyword_objects[i:i + batch_size]
        db.bulk_save_objects(batch)
        saved_count += len(batch)
        print(f"Repository: Saved batch of {len(batch)} keywords ({saved_count}/{len(new_keywords)} total).")

    # Mark batch as processed
    mark_batch_processed(db, domain_id, batch_hash, saved_count)

    print(f"Repository: Successfully saved {saved_count} new keywords from batch {batch_hash}.")
    return saved_count

def mark_batch_processed(db: Session, domain_id: int, batch_hash: str, keyword_count: int):
    """
    Mark a batch as processed in the database.
    """
    try:
        batch_record = models.KeywordBatch(
            domain_id=domain_id,
            batch_hash=batch_hash,
            keyword_count=keyword_count,
            processed=True
        )
        db.add(batch_record)
        db.commit()
        print(f"Repository: Marked batch {batch_hash} as processed with {keyword_count} keywords.")
    except Exception as e:
        print(f"Repository: Failed to mark batch {batch_hash} as processed: {e}")
        db.rollback()
        raise

def get_processed_batches(db: Session, domain_id: int) -> set:
    """
    Get set of already processed batch hashes for a domain.
    """
    try:
        batches = db.query(models.KeywordBatch.batch_hash).filter(
            models.KeywordBatch.domain_id == domain_id,
            models.KeywordBatch.processed == True
        ).all()
        return {batch.batch_hash for batch in batches}
    except Exception as e:
        print(f"Repository: Failed to get processed batches: {e}")
        return set()

def update_seed_keywords_status(db: Session, domain_id: int, seed_keywords: list[str], status: models.SeedStatus):
    """
    Update the seed_status for a list of seed keywords in a domain.
    """
    if not seed_keywords:
        print("Repository: No seed keywords provided to update status.")
        return

    print(f"Repository: Updating {len(seed_keywords)} seed keywords to status '{status.value}' for domain ID {domain_id}.")

    # Update seed keywords status and processed_at timestamp
    db.query(models.Keyword).filter(
        models.Keyword.domain_id == domain_id,
        models.Keyword.text.in_(seed_keywords),
        models.Keyword.keyword_type == models.KeywordType.SEED_ONLY
    ).update({
        'seed_status': status,
        'processed_at': func.now()
    })

    db.commit()
    print(f"Repository: Successfully updated status for {len(seed_keywords)} seed keywords.")

def get_seed_keywords_with_status(db: Session, domain_id: int):
    """
    Get all seed keywords for a domain with their current status.
    """
    print(f"Repository: Retrieving seed keywords with status for domain ID {domain_id}.")
    seed_keywords = db.query(models.Keyword).filter(
        models.Keyword.domain_id == domain_id,
        models.Keyword.keyword_type == models.KeywordType.SEED_ONLY
    ).all()

    return [{
        'id': kw.id,
        'text': kw.text,
        'status': kw.seed_status.value,
        'processed_at': kw.processed_at
    } for kw in seed_keywords]

def get_domain_with_keywords(db: Session, domain_id: int) -> models.Domain | None:
    """
    Retrieves a domain and all its associated keywords using an efficient join.
    """
    print(f"Repository: Retrieving domain ID {domain_id} with all its keywords.")
    return db.query(models.Domain).options(joinedload(models.Domain.keywords)).filter(models.Domain.id == domain_id).first()

def update_domain_status_and_count(db: Session, domain_id: int, status: models.JobStatus):
    """
    Updates domain status and recalculates keyword count.
    """
    print(f"Repository: Updating domain ID {domain_id} status to {status.value}.")

    # Get keyword count
    keyword_count = db.query(models.Keyword).filter(models.Keyword.domain_id == domain_id).count()

    # Update domain
    db.query(models.Domain).filter(models.Domain.id == domain_id).update({
        'status': status,
        'keyword_count': keyword_count
    })

    db.commit()
    print(f"Repository: Domain updated with status {status.value} and {keyword_count} keywords.")

def create_seed_keywords_for_domain_conditional(db: Session, domain_id: int, seed_keywords: list[str]):
    """
    Only saves seed keywords that don't already exist for this domain.
    Prevents duplicate keywords on re-runs.
    Uses bulk operations for better performance with large datasets.
    """
    if not seed_keywords:
        print("Repository: No seed keywords provided to insert.")
        return

    # Check existing keywords for this domain
    existing_texts = set()
    existing_keywords = db.query(models.Keyword).filter(models.Keyword.domain_id == domain_id).all()
    existing_texts = {kw.text for kw in existing_keywords}

    # Filter out keywords that already exist
    new_keywords = [kw for kw in seed_keywords if kw not in existing_texts]

    if not new_keywords:
        print(f"Repository: All {len(seed_keywords)} seed keywords already exist for domain {domain_id}. Skipping.")
        return

    print(f"Repository: Saving {len(new_keywords)} new seed keywords out of {len(seed_keywords)} for domain {domain_id}.")

    # Use bulk operations for better performance
    keyword_objects = []
    for keyword_text in new_keywords:
        keyword_objects.append(models.Keyword(
            text=keyword_text,
            keyword_type=models.KeywordType.SEED_ONLY,
            domain_id=domain_id
        ))

    # Bulk save in batches to handle large datasets efficiently
    batch_size = 1000
    saved_count = 0

    for i in range(0, len(keyword_objects), batch_size):
        batch = keyword_objects[i:i + batch_size]
        db.bulk_save_objects(batch)

        # Commit every batch to reduce memory pressure
        db.commit()

        saved_count += len(batch)
        print(f"Repository: Committed batch of {len(batch)} keywords ({saved_count}/{len(new_keywords)} total).")

    print(f"Repository: Successfully saved {len(new_keywords)} new seed keywords.")

def get_domain_with_keywords(db: Session, domain_id: int) -> models.Domain | None:
    """
    Retrieves a domain and all its associated keywords using an efficient join.
    """
    print(f"Repository: Retrieving domain ID {domain_id} with all its keywords.")
    return db.query(models.Domain).options(joinedload(models.Domain.keywords)).filter(models.Domain.id == domain_id).first()
