"""
Section detection component for clinical documents.
Identifies and labels different sections in clinical notes.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Pattern, Tuple
from pathlib import Path

from ..core.pipeline import PipelineComponent
from ..core.document import Document
from ..core.annotation import Annotation, Span


logger = logging.getLogger(__name__)


class SectionDetector(PipelineComponent):
    """
    Clinical document section detection component.
    Identifies sections like History, Physical Exam, Assessment, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("section_detector", config)
        
        # Section patterns
        self.section_patterns = self.config.get('section_patterns', {})
        self.case_sensitive = self.config.get('case_sensitive', False)
        
        # Default clinical sections
        self.default_sections = self._get_default_sections()
        
        # Compiled patterns
        self.compiled_patterns = {}
        
        # Section hierarchy
        self.section_hierarchy = self.config.get('section_hierarchy', {})
    
    def _get_default_sections(self) -> Dict[str, List[str]]:
        """Get default clinical section patterns."""
        return {
            'CHIEF_COMPLAINT': [
                r'\b(?:chief\s+complaint|cc|presenting\s+complaint)\s*:',
                r'\bcc\s*:',
                r'\bchief\s+complaint\s*:'
            ],
            'HISTORY_PRESENT_ILLNESS': [
                r'\b(?:history\s+of\s+present\s+illness|hpi)\s*:',
                r'\bhpi\s*:',
                r'\bpresent\s+illness\s*:'
            ],
            'PAST_MEDICAL_HISTORY': [
                r'\b(?:past\s+medical\s+history|pmh|medical\s+history)\s*:',
                r'\bpmh\s*:',
                r'\bpast\s+history\s*:'
            ],
            'FAMILY_HISTORY': [
                r'\b(?:family\s+history|fh)\s*:',
                r'\bfh\s*:',
                r'\bfamily\s+hx\s*:'
            ],
            'SOCIAL_HISTORY': [
                r'\b(?:social\s+history|sh)\s*:',
                r'\bsh\s*:',
                r'\bsocial\s+hx\s*:'
            ],
            'ALLERGIES': [
                r'\b(?:allergies|allergy|nkda)\s*:',
                r'\bnkda\s*:',
                r'\bno\s+known\s+drug\s+allergies\s*:'
            ],
            'MEDICATIONS': [
                r'\b(?:medications|meds|current\s+medications)\s*:',
                r'\bmeds\s*:',
                r'\bcurrent\s+meds\s*:'
            ],
            'REVIEW_OF_SYSTEMS': [
                r'\b(?:review\s+of\s+systems|ros)\s*:',
                r'\bros\s*:',
                r'\bsystems\s+review\s*:'
            ],
            'PHYSICAL_EXAM': [
                r'\b(?:physical\s+exam(?:ination)?|pe)\s*:',
                r'\bpe\s*:',
                r'\bexam(?:ination)?\s*:'
            ],
            'VITAL_SIGNS': [
                r'\b(?:vital\s+signs|vitals|vs)\s*:',
                r'\bvs\s*:',
                r'\bvitals\s*:'
            ],
            'LABORATORY': [
                r'\b(?:laboratory|lab\s+(?:results|values|data)|labs)\s*:',
                r'\blabs\s*:',
                r'\blab\s+results\s*:'
            ],
            'IMAGING': [
                r'\b(?:imaging|radiology|x-ray|ct|mri|ultrasound)\s*:',
                r'\bradiology\s*:',
                r'\bimaging\s+studies\s*:'
            ],
            'ASSESSMENT': [
                r'\b(?:assessment|impression|diagnosis)\s*:',
                r'\bassessment\s+and\s+plan\s*:',
                r'\bimpression\s*:'
            ],
            'PLAN': [
                r'\b(?:plan|treatment\s+plan)\s*:',
                r'\bplan\s*:',
                r'\btreatment\s*:'
            ],
            'DISCHARGE_SUMMARY': [
                r'\b(?:discharge\s+summary|hospital\s+course)\s*:',
                r'\bdischarge\s+instructions\s*:',
                r'\bhospital\s+course\s*:'
            ],
            'PROCEDURES': [
                r'\b(?:procedures|interventions)\s*:',
                r'\bprocedures\s+performed\s*:',
                r'\binterventions\s*:'
            ]
        }
    
    async def initialize(self):
        """Initialize section detector."""
        logger.info(f"Initializing {self.name}")
        
        # Merge default and custom sections
        all_sections = {**self.default_sections, **self.section_patterns}
        
        # Compile patterns
        flags = 0 if self.case_sensitive else re.IGNORECASE
        
        for section_name, patterns in all_sections.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_pattern = re.compile(pattern, flags | re.MULTILINE)
                    compiled_patterns.append(compiled_pattern)
                except re.error as e:
                    logger.error(f"Invalid section pattern for {section_name}: {e}")
            
            if compiled_patterns:
                self.compiled_patterns[section_name] = compiled_patterns
        
        logger.info(f"Compiled patterns for {len(self.compiled_patterns)} sections")
    
    async def process(self, document: Document) -> Document:
        """Detect sections in document."""
        logger.debug(f"Processing document with {self.name}")
        
        # Find all section headers
        sections = self._find_sections(document.text)
        
        # Determine section boundaries
        section_boundaries = self._determine_boundaries(sections, len(document.text))
        
        # Create section annotations
        for i, section in enumerate(section_boundaries):
            annotation = Annotation(
                id=f"section_{section['type']}_{section['start']}_{section['end']}",
                span=Span(section['start'], section['end'], document.text[section['start']:section['end']]),
                label='SECTION',
                annotation_type='section',
                confidence=1.0,
                annotator=self.name,
                attributes={
                    'section_type': section['type'],
                    'header_text': section['header_text'],
                    'header_span': section['header_span']
                }
            )
            document.add_annotation(annotation)
        
        return document
    
    def _find_sections(self, text: str) -> List[Dict[str, Any]]:
        """Find all section headers in text."""
        sections = []
        
        for section_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    section = {
                        'type': section_type,
                        'start': match.start(),
                        'end': match.end(),
                        'header_text': match.group(),
                        'header_span': Span(match.start(), match.end(), match.group())
                    }
                    sections.append(section)
        
        # Sort by position
        sections.sort(key=lambda x: x['start'])
        
        # Remove duplicates and overlaps
        sections = self._remove_overlapping_sections(sections)
        
        return sections
    
    def _remove_overlapping_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping section headers."""
        if not sections:
            return sections
        
        filtered_sections = [sections[0]]
        
        for section in sections[1:]:
            last_section = filtered_sections[-1]
            
            # Check for overlap
            if section['start'] < last_section['end']:
                # Keep the longer match
                if (section['end'] - section['start']) > (last_section['end'] - last_section['start']):
                    filtered_sections[-1] = section
            else:
                filtered_sections.append(section)
        
        return filtered_sections
    
    def _determine_boundaries(self, sections: List[Dict[str, Any]], text_length: int) -> List[Dict[str, Any]]:
        """Determine section content boundaries."""
        if not sections:
            return []
        
        section_boundaries = []
        
        for i, section in enumerate(sections):
            # Section starts after the header
            content_start = section['end']
            
            # Section ends at the start of the next section or end of document
            if i + 1 < len(sections):
                content_end = sections[i + 1]['start']
            else:
                content_end = text_length
            
            # Create section with content
            section_with_content = {
                'type': section['type'],
                'start': content_start,
                'end': content_end,
                'header_text': section['header_text'],
                'header_span': section['header_span']
            }
            section_boundaries.append(section_with_content)
        
        return section_boundaries
    
    def get_section_content(self, document: Document, section_type: str) -> List[str]:
        """Get content of specific section type."""
        section_contents = []
        
        for annotation in document.annotations:
            if (annotation.label == 'SECTION' and 
                annotation.attributes.get('section_type') == section_type):
                section_contents.append(annotation.text)
        
        return section_contents
    
    def get_all_sections(self, document: Document) -> Dict[str, List[str]]:
        """Get all sections organized by type."""
        sections = {}
        
        for annotation in document.annotations:
            if annotation.label == 'SECTION':
                section_type = annotation.attributes.get('section_type')
                if section_type:
                    if section_type not in sections:
                        sections[section_type] = []
                    sections[section_type].append(annotation.text)
        
        return sections
    
    def add_section_pattern(self, section_type: str, pattern: str):
        """Add a new section pattern."""
        if section_type not in self.section_patterns:
            self.section_patterns[section_type] = []
        
        self.section_patterns[section_type].append(pattern)
        
        # Recompile patterns
        flags = 0 if self.case_sensitive else re.IGNORECASE
        try:
            compiled_pattern = re.compile(pattern, flags | re.MULTILINE)
            
            if section_type not in self.compiled_patterns:
                self.compiled_patterns[section_type] = []
            
            self.compiled_patterns[section_type].append(compiled_pattern)
            
        except re.error as e:
            logger.error(f"Invalid section pattern: {e}")
    
    def get_section_stats(self) -> Dict[str, Any]:
        """Get section detection statistics."""
        return {
            'total_section_types': len(self.compiled_patterns),
            'total_patterns': sum(len(patterns) for patterns in self.compiled_patterns.values()),
            'section_types': list(self.compiled_patterns.keys())
        }
