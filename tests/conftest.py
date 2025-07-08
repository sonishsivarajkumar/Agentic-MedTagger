"""
Test configuration and fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_clinical_text():
    """Sample clinical text for testing."""
    return """
CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS: 
The patient is a 65-year-old male with a history of hypertension and diabetes mellitus 
who presents with acute onset chest pain and dyspnea. The pain started 2 hours ago 
while the patient was at rest. The pain is described as crushing and substernal. 
The patient denies any associated nausea, vomiting, or diaphoresis.

PAST MEDICAL HISTORY:
1. Hypertension
2. Type 2 diabetes mellitus
3. Hyperlipidemia

MEDICATIONS:
1. Lisinopril 10mg daily
2. Metformin 1000mg twice daily
3. Atorvastatin 20mg daily

ASSESSMENT AND PLAN:
1. Acute coronary syndrome - obtain EKG, cardiac enzymes, chest X-ray
2. Continue current medications
3. Admit to cardiac care unit for monitoring
"""


@pytest.fixture
def sample_document(sample_clinical_text):
    """Sample document for testing."""
    from agentic_medtagger.core.document import Document, DocumentMetadata
    
    metadata = DocumentMetadata(
        document_id="test_doc_001",
        document_type="progress_note",
        patient_id="PAT123456",
        encounter_id="ENC789012"
    )
    
    return Document(text=sample_clinical_text, metadata=metadata)
