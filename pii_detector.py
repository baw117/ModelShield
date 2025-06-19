
"""
Enhanced PII Detection using Microsoft Presidio
Replaces regex-based detection with context-aware NLP models
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum


from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer
from guardrails_server import GuardrailViolation, ViolationType
from guardrails_server import ComprehensivePatternLibrary, ContextAnalyzer
logger = logging.getLogger(__name__)

@dataclass
class PiiDetection:
    """Enhanced PII detection result with confidence and context"""
    entity_type: str
    start: int
    end: int
    score: float
    text: str
    context: Optional[str] = None
    is_validated: bool = False
class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
class EnhancedPiiDetector:
    """Context-aware PII detector using Microsoft Presidio"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.analyzer = None
        self.anonymizer = None
        self._initialize_presidio()
    
    def _initialize_presidio(self):
        """Initialize Presidio with custom configuration"""
        try:
            # Configure NLP engine (using spaCy by default)
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            }
            
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
            
            # Create analyzer with custom recognizers
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            
            # Add custom recognizers
            self._add_custom_recognizers(registry)
            
            # Initialize analyzer and anonymizer
            self.analyzer = AnalyzerEngine(
                registry=registry,
                nlp_engine=nlp_engine,
                supported_languages=["en"]
            )
            self.anonymizer = AnonymizerEngine()
            
            logger.info("Presidio PII detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Presidio: {e}")
            # Fallback to basic regex if Presidio fails
            self._initialize_fallback()
    
    def _add_custom_recognizers(self, registry: RecognizerRegistry):
        """Add custom PII recognizers for specific domains"""
        
        # Enhanced credit card recognizer with validation
        credit_card_patterns = [
            Pattern("Visa", r"\b4[0-9]{12}(?:[0-9]{3})?\b", 0.9),
            Pattern("MasterCard", r"\b5[1-5][0-9]{14}\b", 0.9),
            Pattern("Amex", r"\b3[47][0-9]{13}\b", 0.9),
            Pattern("Discover", r"\b6(?:011|5[0-9]{2})[0-9]{12}\b", 0.9),
        ]
        
        credit_card_recognizer = PatternRecognizer(
            supported_entity="CREDIT_CARD_ENHANCED",
            patterns=credit_card_patterns,
            context=["card", "credit", "payment", "billing"]
        )
        registry.add_recognizer(credit_card_recognizer)
        
        # Enhanced SSN recognizer with context
        ssn_patterns = [
            Pattern("SSN_FULL", r"\b\d{3}-\d{2}-\d{4}\b", 0.9),
            Pattern("SSN_SPACES", r"\b\d{3}\s\d{2}\s\d{4}\b", 0.8),
            Pattern("SSN_DOTS", r"\b\d{3}\.\d{2}\.\d{4}\b", 0.8),
        ]
        
        ssn_recognizer = PatternRecognizer(
            supported_entity="SSN_ENHANCED",
            patterns=ssn_patterns,
            context=["ssn", "social", "security", "tax", "employee"]
        )
        registry.add_recognizer(ssn_recognizer)
        
        # API Keys and Tokens
        api_key_patterns = [
            Pattern("AWS_ACCESS_KEY", r"\bAKIA[0-9A-Z]{16}\b", 0.9),
            Pattern("GITHUB_TOKEN", r"\bghp_[0-9a-zA-Z]{36}\b", 0.9),
            Pattern("OPENAI_KEY", r"\bsk-[0-9a-zA-Z]{48}\b", 0.9),
            Pattern("GENERIC_API_KEY", r"\b[0-9a-fA-F]{32}\b", 0.6),
        ]
        
        api_key_recognizer = PatternRecognizer(
            supported_entity="API_KEY",
            patterns=api_key_patterns,
            context=["api", "key", "token", "secret", "auth"]
        )
        registry.add_recognizer(api_key_recognizer)
    
    def _initialize_fallback(self):
        """Initialize fallback regex-based detection if Presidio fails"""
        logger.warning("Using fallback regex-based PII detection")
        self.fallback_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_us': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        }
    
    def detect_pii(self, text: str) -> List[PiiDetection]:
        """Detect PII with context awareness and confidence scoring"""
        detections = []
        
        if self.analyzer:
            try:
                # Use Presidio for detection
                presidio_results = self.analyzer.analyze(
                    text=text,
                    language="en",
                    score_threshold=self.confidence_threshold
                )
                
                for result in presidio_results:
                    # Extract the actual text
                    detected_text = text[result.start:result.end]
                    
                    # Validate the detection if possible
                    is_validated = self._validate_entity(result.entity_type, detected_text)
                    
                    # Get surrounding context
                    context = self._get_context(text, result.start, result.end)
                    
                    detection = PiiDetection(
                        entity_type=result.entity_type,
                        start=result.start,
                        end=result.end,
                        score=result.score,
                        text=detected_text,
                        context=context,
                        is_validated=is_validated
                    )
                    detections.append(detection)
                
            except Exception as e:
                logger.error(f"Presidio detection failed: {e}")
                # Fall back to regex
                detections = self._fallback_detection(text)
        else:
            # Use fallback detection
            detections = self._fallback_detection(text)
        
        return detections
    
    def _validate_entity(self, entity_type: str, text: str) -> bool:
        """Validate detected entities using algorithms like Luhn"""
        if entity_type in ["CREDIT_CARD", "CREDIT_CARD_ENHANCED"]:
            return self._validate_credit_card(text)
        elif entity_type in ["SSN", "SSN_ENHANCED"]:
            return self._validate_ssn(text)
        elif entity_type == "EMAIL_ADDRESS":
            return "@" in text and "." in text.split("@")[-1]
        return True  # Default to valid for other types
    
    def _validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm"""

        digits = ''.join(filter(str.isdigit, card_number))
        
        if len(digits) < 13 or len(digits) > 19:
            return False
        

        total = 0
        reverse_digits = digits[::-1]
        
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:  # Every second digit from right
                n *= 2
                if n > 9:
                    n = n // 10 + n % 10
            total += n
        
        return total % 10 == 0
    
    def _validate_ssn(self, ssn: str) -> bool:
        """Basic SSN validation (format and known invalid patterns)"""

        digits = ''.join(filter(str.isdigit, ssn))
        
        if len(digits) != 9:
            return False
        

        invalid_patterns = [
            "000000000", "111111111", "222222222", "333333333",
            "444444444", "555555555", "666666666", "777777777",
            "888888888", "999999999"
        ]
        
        if digits in invalid_patterns:
            return False
        

        area = digits[:3]
        if area in ["000", "666"] or area.startswith("9"):
            return False
        
        return True
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for better understanding"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end]

        masked_context = context[:start-context_start] + "[MASKED]" + context[end-context_start:]
        
        return masked_context.strip()
    
    def _fallback_detection(self, text: str) -> List[PiiDetection]:
        """Fallback regex-based detection"""
        import re
        detections = []
        
        for entity_type, pattern in self.fallback_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                detection = PiiDetection(
                    entity_type=entity_type.upper(),
                    start=match.start(),
                    end=match.end(),
                    score=0.8,  # Default confidence for regex
                    text=match.group(),
                    context=self._get_context(text, match.start(), match.end()),
                    is_validated=False
                )
                detections.append(detection)
        
        return detections
    
    def anonymize_text(self, text: str, detections: List[PiiDetection]) -> str:
        """Anonymize detected PII in text"""
        if not self.anonymizer or not detections:
            return text
        
        try:

            from presidio_analyzer import RecognizerResult
            
            presidio_results = []
            for detection in detections:
                result = RecognizerResult(
                    entity_type=detection.entity_type,
                    start=detection.start,
                    end=detection.end,
                    score=detection.score
                )
                presidio_results.append(result)
            

            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=presidio_results
            )
            
            return anonymized_result.text
            
        except Exception as e:
            logger.error(f"Anonymization failed: {e}")
            return text
    
    def get_detection_summary(self, detections: List[PiiDetection]) -> Dict:
        """Get summary statistics for detections"""
        if not detections:
            return {"total": 0, "types": {}, "high_confidence": 0, "validated": 0}
        
        summary = {
            "total": len(detections),
            "types": {},
            "high_confidence": 0,
            "validated": 0,
            "average_confidence": 0.0
        }
        
        total_confidence = 0
        for detection in detections:

            entity_type = detection.entity_type
            if entity_type not in summary["types"]:
                summary["types"][entity_type] = 0
            summary["types"][entity_type] += 1
            

            if detection.score >= 0.8:
                summary["high_confidence"] += 1
            

            if detection.is_validated:
                summary["validated"] += 1
            
            total_confidence += detection.score
        
        summary["average_confidence"] = total_confidence / len(detections)
        
        return summary


def integrate_with_existing_detector():
    """Example of how to integrate with the existing ComprehensiveDetector"""
    
    class EnhancedComprehensiveDetector:
        def __init__(self, config):

            self.config = config
            self.patterns = ComprehensivePatternLibrary()
            self.context_analyzer = ContextAnalyzer(self.patterns)
            self._compile_patterns()
            

            self.pii_detector = EnhancedPiiDetector(confidence_threshold=0.7)
        
        def _detect_pii_enhanced(self, text: str) -> List[GuardrailViolation]:
            """Enhanced PII detection using Presidio"""
            violations = []
            
            # Get PII detections
            detections = self.pii_detector.detect_pii(text)
            
            if detections:

                detections_by_type = {}
                for detection in detections:
                    if detection.entity_type not in detections_by_type:
                        detections_by_type[detection.entity_type] = []
                    detections_by_type[detection.entity_type].append(detection)
                

                for entity_type, entity_detections in detections_by_type.items():

                    severity = self._get_pii_severity(entity_type, entity_detections)
                    

                    avg_confidence = sum(d.score for d in entity_detections) / len(entity_detections)
                    

                    detection_summary = f"{len(entity_detections)} instances"
                    if any(d.is_validated for d in entity_detections):
                        detection_summary += f" ({sum(1 for d in entity_detections if d.is_validated)} validated)"
                    
                    violation = GuardrailViolation(
                        violation_type=ViolationType.PII_DETECTED,
                        severity=severity,
                        message=f"Enhanced PII detected: {entity_type} ({detection_summary})",
                        pattern_matched=f"Presidio:{entity_type}",
                        confidence_score=avg_confidence,
                        suggested_action=self._get_pii_action(entity_type, severity),
                        location=f"Context: {entity_detections[0].context[:50]}..." if entity_detections[0].context else None
                    )
                    violations.append(violation)
            
            return violations
        
        def _get_pii_severity(self, entity_type: str, detections: List[PiiDetection]) -> Severity:
            """Determine severity based on PII type and context"""

            if entity_type in ["SSN", "SSN_ENHANCED", "CREDIT_CARD", "CREDIT_CARD_ENHANCED"]:
                return Severity.CRITICAL
            

            if entity_type in ["API_KEY", "PASSWORD", "CRYPTO"]:
                return Severity.HIGH
            

            if entity_type in ["PHONE_NUMBER", "EMAIL_ADDRESS"]:

                if any(d.is_validated for d in detections):
                    return Severity.MEDIUM
                else:
                    return Severity.LOW
            

            return Severity.MEDIUM
        
        def _get_pii_action(self, entity_type: str, severity: Severity) -> str:
            """Get suggested action for PII type"""
            if severity == Severity.CRITICAL:
                return "Block immediately and log security incident"
            elif severity == Severity.HIGH:
                return "Block and anonymize"
            elif severity == Severity.MEDIUM:
                return "Anonymize or redact"
            else:
                return "Monitor and potentially redact"


if __name__ == "__main__":

    detector = EnhancedPiiDetector()
    
    test_texts = [
        "My SSN is 123-45-6789 and my email is john@example.com",
        "Call me at (555) 123-4567 or use card 4532-1234-5678-9012",
        "API key: sk-1234567890abcdef1234567890abcdef12345678",
        "The number 1234567890 could be a phone or something else",
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1} ---")
        print(f"Text: {text}")
        
        detections = detector.detect_pii(text)
        summary = detector.get_detection_summary(detections)
        
        print(f"Summary: {summary}")
        
        for detection in detections:
            print(f"  - {detection.entity_type}: '{detection.text}' "
                  f"(confidence: {detection.score:.2f}, validated: {detection.is_validated})")
            if detection.context:
                print(f"    Context: {detection.context}")
        
        if detections:
            anonymized = detector.anonymize_text(text, detections)
            print(f"Anonymized: {anonymized}")