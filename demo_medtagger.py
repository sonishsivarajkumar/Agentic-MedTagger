#!/usr/bin/env python3
"""
Demo script for Agentic MedTagger - showcasing all Mayo Clinic MedTagger features.
"""

import asyncio
import json
from pathlib import Path
import sys
import time

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic_medtagger.core.document import Document, DocumentMetadata
from agentic_medtagger.core.pipeline import create_medtagger_pipeline
from config.medtagger_config import default_config, minimal_config


def create_sample_clinical_documents():
    """Create sample clinical documents for testing."""
    documents = []
    
    # Document 1: Emergency Department Note
    doc1_text = """
    CHIEF COMPLAINT: Chest pain and shortness of breath
    
    HISTORY OF PRESENT ILLNESS: 
    This is a 65-year-old male with a history of diabetes mellitus and hypertension 
    who presents to the emergency department with acute onset chest pain that started 
    2 hours ago. The pain is described as crushing, substernal, and radiates to the 
    left arm. Patient also reports associated shortness of breath and diaphoresis.
    Patient denies any recent trauma or similar episodes in the past.
    
    PAST MEDICAL HISTORY:
    1. Type 2 diabetes mellitus - diagnosed 10 years ago
    2. Hypertension - well controlled on lisinopril
    3. Hyperlipidemia - on atorvastatin
    
    MEDICATIONS:
    1. Metformin 1000mg twice daily
    2. Lisinopril 10mg daily  
    3. Atorvastatin 20mg daily
    4. Aspirin 81mg daily
    
    ALLERGIES: No known drug allergies
    
    SOCIAL HISTORY: Former smoker, quit 5 years ago. Drinks alcohol occasionally.
    
    PHYSICAL EXAMINATION:
    Vital signs: Blood pressure 160/95 mmHg, heart rate 102 bpm, temperature 98.6°F, 
    respiratory rate 22/min, oxygen saturation 96% on room air.
    
    General: Patient appears anxious and diaphoretic
    Cardiovascular: Tachycardic, regular rhythm, no murmurs
    Pulmonary: Mild bilateral rales at bases
    
    LABORATORY:
    Troponin I: 2.5 ng/mL (elevated)
    Creatinine: 1.2 mg/dL
    Glucose: 180 mg/dL
    
    ASSESSMENT AND PLAN:
    1. Acute myocardial infarction - likely STEMI
       - Start dual antiplatelet therapy
       - Consider cardiac catheterization
    2. Diabetes mellitus - continue home medications
    3. Hypertension - may need adjustment given current readings
    """
    
    documents.append(Document(
        text=doc1_text,
        document_id="ed_note_001",
        metadata=DocumentMetadata(
            document_id="ed_note_001",
            document_type="emergency_department", 
            patient_id="12345"
        )
    ))
    
    # Document 2: Discharge Summary
    doc2_text = """
    DISCHARGE SUMMARY
    
    PATIENT: Jane Smith
    DATE OF ADMISSION: 01/15/2024
    DATE OF DISCHARGE: 01/18/2024
    
    ADMITTING DIAGNOSIS: Pneumonia
    DISCHARGE DIAGNOSIS: Community-acquired pneumonia, resolved
    
    HOSPITAL COURSE:
    Ms. Smith is a 78-year-old female who was admitted with fever, cough, and 
    shortness of breath. Chest X-ray showed right lower lobe infiltrate consistent 
    with pneumonia. Patient was started on ceftriaxone and azithromycin. 
    
    Patient showed marked improvement over the course of hospitalization. 
    Fever resolved by hospital day 2. Respiratory symptoms improved significantly.
    Follow-up chest X-ray showed clearing of infiltrate.
    
    PAST MEDICAL HISTORY:
    1. Chronic obstructive pulmonary disease
    2. Osteoporosis
    3. Depression - stable on sertraline
    
    DISCHARGE MEDICATIONS:
    1. Amoxicillin 500mg three times daily for 5 days
    2. Albuterol inhaler as needed for shortness of breath
    3. Continue home medications: sertraline, calcium, vitamin D
    
    ALLERGIES: Penicillin - causes rash
    
    FOLLOW-UP:
    Patient should follow up with primary care physician in 1 week.
    Repeat chest X-ray in 4-6 weeks to ensure complete resolution.
    """
    
    documents.append(Document(
        text=doc2_text,
        document_id="discharge_summary_001",
        metadata=DocumentMetadata(
            document_id="discharge_summary_001",
            document_type="discharge_summary", 
            patient_id="67890"
        )
    ))
    
    return documents


async def demo_medtagger_pipeline():
    """Demonstrate the MedTagger pipeline with all features."""
    print("🏥 Agentic MedTagger Demo - Mayo Clinic MedTagger Features")
    print("=" * 60)
    
    # Create sample documents
    print("\n📋 Creating sample clinical documents...")
    documents = create_sample_clinical_documents()
    print(f"Created {len(documents)} sample documents")
    
    # Create pipeline with default configuration
    print("\n🔧 Initializing MedTagger pipeline...")
    pipeline = create_medtagger_pipeline(default_config)
    print(f"Pipeline created with {len(pipeline.components)} components:")
    for component in pipeline.components:
        print(f"  - {component.name}")
    
    # Process documents
    print("\n🔍 Processing documents through MedTagger pipeline...")
    start_time = time.time()
    
    for i, document in enumerate(documents, 1):
        print(f"\n📄 Processing Document {i}: {document.document_id}")
        
        try:
            processed_doc = await pipeline.process_document(document)
            
            # Display results
            print(f"✅ Successfully processed document {document.document_id}")
            print(f"📊 Found {len(processed_doc.annotations)} annotations")
            
            # Group annotations by type
            annotation_types = {}
            for annotation in processed_doc.annotations:
                if annotation.label not in annotation_types:
                    annotation_types[annotation.label] = []
                annotation_types[annotation.label].append(annotation)
            
            print(f"🏷️  Annotation types found:")
            for label, annotations in annotation_types.items():
                print(f"   - {label}: {len(annotations)} items")
                
                # Show a few examples
                for j, annotation in enumerate(annotations[:3]):
                    print(f"     • {annotation.text} (confidence: {annotation.confidence:.2f})")
                
                if len(annotations) > 3:
                    print(f"     ... and {len(annotations) - 3} more")
            
            # Show section detection results
            section_annotations = [a for a in processed_doc.annotations if a.label == 'SECTION']
            if section_annotations:
                print(f"\n📑 Detected {len(section_annotations)} sections:")
                for section in section_annotations:
                    section_type = section.attributes.get('section_type', 'Unknown')
                    print(f"   - {section_type}")
            
            # Show pattern extraction results
            pattern_annotations = [a for a in processed_doc.annotations 
                                 if a.annotator == 'medtagger_ie']
            if pattern_annotations:
                print(f"\n🔍 Pattern extraction found {len(pattern_annotations)} items:")
                for pattern in pattern_annotations[:5]:  # Show first 5
                    print(f"   - {pattern.text} ({pattern.label})")
            
            # Show normalization results
            normalized_annotations = [a for a in processed_doc.annotations 
                                    if 'normalization' in a.attributes]
            if normalized_annotations:
                print(f"\n🔗 Normalized {len(normalized_annotations)} concepts:")
                for norm in normalized_annotations[:3]:  # Show first 3
                    concepts = norm.attributes['normalization'].get('concepts', [])
                    if concepts:
                        concept = concepts[0]  # Show first concept
                        print(f"   - {norm.text} → {concept.get('concept_name', 'Unknown')} "
                              f"({concept.get('vocabulary', 'Unknown')})")
            
        except Exception as e:
            print(f"❌ Error processing document {document.document_id}: {e}")
            import traceback
            traceback.print_exc()
    
    processing_time = time.time() - start_time
    print(f"\n⏱️  Total processing time: {processing_time:.2f} seconds")
    
    # Show pipeline metrics
    print("\n📈 Pipeline Metrics:")
    metrics = pipeline.get_metrics()
    print(f"   - Documents processed: {metrics['documents_processed']}")
    print(f"   - Average processing time: {metrics['average_processing_time']:.3f}s")
    print(f"   - Component average times:")
    for comp_name, avg_time in metrics['component_average_times'].items():
        print(f"     • {comp_name}: {avg_time:.3f}s")
    
    if metrics['error_count'] > 0:
        print(f"   - Errors: {metrics['error_count']}")


async def demo_individual_components():
    """Demonstrate individual MedTagger components."""
    print("\n🧩 Individual Component Demonstration")
    print("=" * 40)
    
    # Test text
    test_text = """
    PHYSICAL EXAM: Blood pressure 140/90 mmHg, heart rate 88 bpm, temperature 98.6°F.
    Patient has diabetes and hypertension. No chest pain or shortness of breath.
    """
    
    print(f"📝 Test text: {test_text.strip()}")
    
    # Test MedTaggerIE (Pattern Extraction)
    print("\n🔍 Testing MedTaggerIE (Pattern Extraction)...")
    from agentic_medtagger.annotators import MedTaggerIE
    
    ie_component = MedTaggerIE({'use_clinical_patterns': True})
    await ie_component.initialize()
    
    test_doc = Document(text=test_text, document_id="test")
    processed_doc = await ie_component.process(test_doc)
    
    ie_annotations = [a for a in processed_doc.annotations if a.annotator == 'medtagger_ie']
    print(f"   Found {len(ie_annotations)} pattern-based annotations:")
    for annotation in ie_annotations:
        print(f"   - {annotation.text} → {annotation.label} (confidence: {annotation.confidence})")
    
    # Test Section Detection
    print("\n📑 Testing Section Detection...")
    from agentic_medtagger.annotators import SectionDetector
    
    section_detector = SectionDetector()
    await section_detector.initialize()
    
    section_doc = Document(text=test_text, document_id="test")
    processed_section_doc = await section_detector.process(section_doc)
    
    section_annotations = [a for a in processed_section_doc.annotations if a.label == 'SECTION']
    print(f"   Found {len(section_annotations)} sections:")
    for section in section_annotations:
        section_type = section.attributes.get('section_type', 'Unknown')
        print(f"   - {section_type}")


def export_results_to_json(documents, output_path="medtagger_results.json"):
    """Export processing results to JSON."""
    results = []
    
    for doc in documents:
        doc_result = {
            "document_id": doc.document_id,
            "document_type": doc.metadata.document_type if hasattr(doc.metadata, 'document_type') else "unknown",
            "text_length": len(doc.text),
            "annotations": []
        }
        
        for annotation in doc.annotations:
            ann_dict = {
                "text": annotation.text,
                "label": annotation.label,
                "start": annotation.start,
                "end": annotation.end,
                "confidence": annotation.confidence,
                "annotator": annotation.annotator,
                "annotation_type": annotation.annotation_type,
                "attributes": annotation.attributes
            }
            doc_result["annotations"].append(ann_dict)
        
        results.append(doc_result)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results exported to {output_path}")


async def main():
    """Main demo function."""
    print("🚀 Starting Agentic MedTagger Demo")
    print("Showcasing all Mayo Clinic MedTagger features in Python")
    print("=" * 60)
    
    try:
        # Run main pipeline demo
        await demo_medtagger_pipeline()
        
        # Run individual component demos
        await demo_individual_components()
        
        print("\n✅ Demo completed successfully!")
        print("\nMedTagger Features Demonstrated:")
        print("  ✓ Section Detection (clinical document structure)")
        print("  ✓ Dictionary-based Matching (flashtext)")
        print("  ✓ Pattern-based Information Extraction (regex + spaCy)")
        print("  ✓ ML-based Named Entity Recognition (spaCy)")
        print("  ✓ OMOP/UMLS Concept Normalization")
        print("  ✓ Comprehensive Pipeline Processing")
        print("  ✓ Performance Metrics and Monitoring")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
