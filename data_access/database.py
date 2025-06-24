from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
import configparser

# Database URL
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    print("Warning: DATABASE_URL environment variable not set. Reading from alembic.ini.")
    config = configparser.ConfigParser()
    config.read("alembic.ini")
    DATABASE_URL = config['alembic']['sqlalchemy.url']

# Create the engine
engine = create_engine(DATABASE_URL)

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
