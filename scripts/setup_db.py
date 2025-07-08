#!/usr/bin/env python3
"""
Database setup script for Agentic MedTagger.
"""

import asyncio
import logging
import os
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()


class Document(Base):
    """Document table."""
    __tablename__ = "documents"
    
    id = sa.Column(sa.String, primary_key=True)
    text = sa.Column(sa.Text, nullable=False)
    metadata_ = sa.Column(sa.JSON, name="metadata")
    created_at = sa.Column(sa.DateTime, server_default=sa.func.now())
    updated_at = sa.Column(sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now())


class Annotation(Base):
    """Annotation table."""
    __tablename__ = "annotations"
    
    id = sa.Column(sa.String, primary_key=True)
    document_id = sa.Column(sa.String, sa.ForeignKey("documents.id"), nullable=False)
    span_start = sa.Column(sa.Integer, nullable=False)
    span_end = sa.Column(sa.Integer, nullable=False)
    span_text = sa.Column(sa.Text, nullable=False)
    label = sa.Column(sa.String, nullable=False)
    annotation_type = sa.Column(sa.String, nullable=False)
    confidence = sa.Column(sa.Float, default=1.0)
    annotator = sa.Column(sa.String, nullable=False)
    attributes = sa.Column(sa.JSON)
    normalized_concept = sa.Column(sa.String)
    umls_cui = sa.Column(sa.String)
    umls_semantic_types = sa.Column(sa.JSON)
    created_at = sa.Column(sa.DateTime, server_default=sa.func.now())


class AnnotationFeedback(Base):
    """Annotation feedback table for active learning."""
    __tablename__ = "annotation_feedback"
    
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    annotation_id = sa.Column(sa.String, sa.ForeignKey("annotations.id"), nullable=False)
    feedback_type = sa.Column(sa.String, nullable=False)  # 'correct', 'incorrect', 'partial'
    corrected_label = sa.Column(sa.String)
    comments = sa.Column(sa.Text)
    user_id = sa.Column(sa.String)
    created_at = sa.Column(sa.DateTime, server_default=sa.func.now())


class ProcessingSession(Base):
    """Processing session tracking."""
    __tablename__ = "processing_sessions"
    
    id = sa.Column(sa.String, primary_key=True)
    pipeline_name = sa.Column(sa.String, nullable=False)
    pipeline_version = sa.Column(sa.String, nullable=False)
    config = sa.Column(sa.JSON)
    started_at = sa.Column(sa.DateTime, server_default=sa.func.now())
    completed_at = sa.Column(sa.DateTime)
    status = sa.Column(sa.String, default="running")  # 'running', 'completed', 'failed'
    error_message = sa.Column(sa.Text)
    documents_processed = sa.Column(sa.Integer, default=0)


async def create_database(database_url: str):
    """Create database and tables."""
    logger.info(f"Creating database with URL: {database_url}")
    
    # Create engine
    engine = create_async_engine(database_url, echo=True)
    
    try:
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise
    
    finally:
        await engine.dispose()


async def setup_database():
    """Setup database from environment configuration."""
    # Get database URL from environment
    database_url = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://postgres:password@localhost:5432/agentic_medtagger"
    )
    
    try:
        await create_database(database_url)
        print("✅ Database setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return 1
    
    return 0


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return asyncio.run(setup_database())


if __name__ == "__main__":
    exit(main())
