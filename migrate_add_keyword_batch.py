#!/usr/bin/env python3
"""
Migration script to add KeywordBatch table for batch tracking.
Run this once to add the new table to your database.
"""

from alembic.config import Config
from alembic import command
import os

def run_migration():
    """Run Alembic migration to add KeywordBatch table"""

    # Set up Alembic configuration
    alembic_cfg = Config("alembic.ini")

    try:
        print("🔄 Running migration to add KeywordBatch table...")

        # Generate new migration
        command.revision(alembic_cfg, message="Add KeywordBatch table for batch tracking", autogenerate=True)

        # Apply the migration
        command.upgrade(alembic_cfg, "head")

        print("✅ Migration completed successfully!")
        print("📋 New table 'keyword_batches' added with columns:")
        print("   - id (Primary Key)")
        print("   - domain_id (Foreign Key to domains)")
        print("   - batch_hash (Unique identifier for batch tracking)")
        print("   - keyword_count (Number of keywords in batch)")
        print("   - processed (Boolean flag)")
        print("   - processed_at (Timestamp)")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("💡 Try running manually:")
        print("   alembic revision --autogenerate -m 'Add KeywordBatch table'")
        print("   alembic upgrade head")

if __name__ == "__main__":
    run_migration()
