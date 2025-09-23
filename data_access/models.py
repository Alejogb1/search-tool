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
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

# Define an Enum for status tracking. This is much better than raw strings.
class JobStatus(enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

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
    
    # This is the foreign key linking back to the Domain table
    domain_id = Column(Integer, ForeignKey("domains.id"), nullable=False)

    # This establishes the other side of the relationship
    domain = relationship("Domain", back_populates="keywords")

    def __repr__(self):
        return f"<Keyword(id={self.id}, text='{self.text}')>"
