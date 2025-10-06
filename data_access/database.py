from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
import configparser

# Database URL
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set. Please configure it in .env file")

# Create the engine with optimized settings for large batch operations
engine = create_engine(
    DATABASE_URL,
    pool_size=20,  # Increase pool size for better concurrency
    max_overflow=30,  # Allow more connections when pool is exhausted
    pool_timeout=30,  # Timeout for getting connection from pool
    pool_recycle=3600,  # Recycle connections after 1 hour
    connect_args={
        "connect_timeout": 60,  # Connection timeout
        "application_name": "MarketIntelligenceAPI",
        "options": "-c statement_timeout=300s"  # 5 minute statement timeout
    } if DATABASE_URL.startswith("postgresql") else {}
)

# Create a SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base for declarative models
Base = declarative_base()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
