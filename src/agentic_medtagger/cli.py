"""
Command-line interface for Agentic MedTagger.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List

from .core.pipeline import Pipeline
from .core.document import Document


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('agentic_medtagger.log')
        ]
    )


async def process_files(file_paths: List[str], config_path: str = None) -> None:
    """Process a list of files."""
    # Load documents
    documents = []
    for file_path in file_paths:
        try:
            doc = Document.from_file(file_path)
            documents.append(doc)
            print(f"Loaded document: {file_path}")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
    
    if not documents:
        print("No documents loaded successfully")
        return
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Add basic components
    from .core.pipeline import TokenizerComponent, DictionaryMatcherComponent
    pipeline.add_component(TokenizerComponent())
    pipeline.add_component(DictionaryMatcherComponent())
    
    # Process documents
    try:
        processed_docs = await pipeline.process_documents(documents)
        
        # Output results
        for doc in processed_docs:
            print(f"\nDocument: {doc.document_id}")
            print(f"Annotations: {len(doc.annotations)}")
            
            # Show annotation summary
            annotation_types = {}
            for ann in doc.annotations:
                ann_type = ann.annotation_type
                if ann_type not in annotation_types:
                    annotation_types[ann_type] = 0
                annotation_types[ann_type] += 1
            
            for ann_type, count in annotation_types.items():
                print(f"  {ann_type}: {count}")
            
            # Output to file
            output_file = f"{doc.document_id}_annotations.json"
            with open(output_file, 'w') as f:
                import json
                json.dump(doc.to_dict(), f, indent=2, default=str)
            print(f"Saved annotations to: {output_file}")
        
        # Show pipeline statistics
        stats = pipeline.get_statistics()
        print(f"\nPipeline Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Processing failed: {e}")
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic MedTagger - Clinical NLP Framework"
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="Input files to process"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Agentic MedTagger 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Change to output directory if specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        import os
        os.chdir(output_path)
    
    # Process files
    try:
        asyncio.run(process_files(args.files, args.config))
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
