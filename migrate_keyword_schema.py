#!/usr/bin/env python3
"""
Migration script to update the keyword schema from the old is_seed integer
to the new KeywordType enum with timestamp fields.

This script should be run once to migrate existing data.
Works with PostgreSQL.
"""

import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_access.models import KeywordType

def migrate_keyword_schema():
    """Migrate existing keyword data to the new schema."""

    # Get database URL from environment
    database_url = "postgresql://postgres.salejjvpcmpmmcepzckk:pYvto9-rytkok-cohxab@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    if not database_url:
        print("Error: DATABASE_URL environment variable not set")
        sys.exit(1)

    print(f"Connecting to database: {database_url[:50]}...")

    # Create engine
    engine = create_engine(database_url)

    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Check if migration is needed (PostgreSQL syntax)
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'keywords' AND table_schema = 'public'
        """))
        columns = [row[0] for row in result.fetchall()]

        if 'keyword_type' in columns:
            print("Migration already completed - keyword_type column exists")
            return

        print("Starting migration...")

        # Step 1: Add new columns
        print("Adding new columns...")
        db.execute(text("ALTER TABLE keywords ADD COLUMN keyword_type VARCHAR(20) DEFAULT 'seed_only'"))
        db.execute(text("ALTER TABLE keywords ADD COLUMN seeded_at TIMESTAMP"))
        db.execute(text("ALTER TABLE keywords ADD COLUMN expanded_at TIMESTAMP"))
        db.commit()

        # Step 2: Migrate existing data
        print("Migrating existing data...")

        # Update seed keywords (is_seed = 1) to SEED_ONLY
        db.execute(text(f"""
            UPDATE keywords
            SET keyword_type = '{KeywordType.SEED_ONLY.value}',
                seeded_at = CURRENT_TIMESTAMP
            WHERE is_seed = 1
        """))

        # Update expanded keywords (is_seed = 0) to SEED_EXPANDED
        # (assuming they were originally seeds that got expanded)
        db.execute(text(f"""
            UPDATE keywords
            SET keyword_type = '{KeywordType.SEED_EXPANDED.value}',
                expanded_at = CURRENT_TIMESTAMP
            WHERE is_seed = 0 AND avg_monthly_searches IS NOT NULL
        """))

        # Update keywords that are expanded but have no search data to EXPANDED_ONLY
        db.execute(text(f"""
            UPDATE keywords
            SET keyword_type = '{KeywordType.EXPANDED_ONLY.value}',
                expanded_at = CURRENT_TIMESTAMP
            WHERE is_seed = 0 AND avg_monthly_searches IS NULL
        """))

        db.commit()

        # Step 3: Remove old column
        print("Removing old is_seed column...")
        db.execute(text("ALTER TABLE keywords DROP COLUMN is_seed"))
        db.commit()

        print("Migration completed successfully!")

        # Verify the migration
        result = db.execute(text("SELECT keyword_type, COUNT(*) as count FROM keywords GROUP BY keyword_type"))
        print("\nMigration results:")
        for row in result:
            print(f"  {row[0]}: {row[1]} keywords")

    except Exception as e:
        print(f"Migration failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    migrate_keyword_schema()
