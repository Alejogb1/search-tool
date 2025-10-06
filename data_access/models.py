# data_access/models.py
import enum
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Enum,
    ForeignKey,
    Boolean
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

# Define an Enum for status tracking. This is much better than raw strings.
class JobStatus(enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

# Define an Enum for keyword types to handle seed vs expanded relationships
class KeywordType(enum.Enum):
    SEED_ONLY = "seed_only"           # Only exists as seed, no expansion data
    SEED_EXPANDED = "seed_expanded"   # Started as seed, got expanded data
    EXPANDED_ONLY = "expanded_only"   # Only found during expansion

Base = declarative_base()

class Domain(Base):
    __tablename__ = "domains"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False, unique=True, index=True)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.PENDING)
    keyword_count = Column(Integer, default=0, nullable=False)  # New keyword count column
    
    # Timestamps are managed by the database server for reliability
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # This creates the one-to-many relationship. 
    # 'keywords' will be a Python list of Keyword objects.
    keywords = relationship("Keyword", back_populates="domain", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Domain(id={self.id}, url='{self.url}', status='{self.status.value}')>"


class Keyword(Base):
    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False, index=True)

    # Pre-emptively add columns for Phase 2 data
    avg_monthly_searches = Column(Integer, nullable=True)
    competition_level = Column(String, nullable=True)

    # New fields for seed vs expanded keyword distinction using enum
    keyword_type = Column(Enum(KeywordType), nullable=False, default=KeywordType.SEED_ONLY)
    batch_info = Column(String, nullable=True)  # Track which batch this keyword came from

    # Timestamps to track keyword lifecycle
    seeded_at = Column(DateTime(timezone=True), nullable=True)    # When it was first seeded
    expanded_at = Column(DateTime(timezone=True), nullable=True)  # When it got expansion data

    # This is the foreign key linking back to the Domain table
    domain_id = Column(Integer, ForeignKey("domains.id"), nullable=False)

    # This establishes the other side of the relationship
    domain = relationship("Domain", back_populates="keywords")

    def __repr__(self):
        return f"<Keyword(id={self.id}, text='{self.text}', type='{self.keyword_type.value}')>"


class KeywordBatch(Base):
    __tablename__ = "keyword_batches"

    id = Column(Integer, primary_key=True, index=True)
    domain_id = Column(Integer, ForeignKey("domains.id"), nullable=False)
    batch_hash = Column(String, nullable=False, index=True)  # Unique hash for batch tracking
    keyword_count = Column(Integer, nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationship back to domain
    domain = relationship("Domain", back_populates="keyword_batches")

    def __repr__(self):
        return f"<KeywordBatch(id={self.id}, domain_id={self.domain_id}, hash='{self.batch_hash}', processed={self.processed})>"


# Add relationship to Domain model
Domain.keyword_batches = relationship("KeywordBatch", back_populates="domain", cascade="all, delete-orphan")
