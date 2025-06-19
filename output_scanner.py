
"""
LLM Output Scanner and Response Guardrails
Validates and filters LLM responses before returning to users
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class OutputViolationType(Enum):
    """Types of violations that can occur in LLM outputs"""
    LEAKED_PII = "leaked_pii"
    HALLUCINATED_FACTS = "hallucinated_facts"
    TOXIC_LANGUAGE = "toxic_language"
    HATE_SPEECH_OUTPUT = "hate_speech_output"
    VIOLENCE_OUTPUT = "violence_output"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    PROMPT_INJECTION_SUCCESS = "prompt_injection_success"
    TRAINING_DATA_LEAK = "training_data_leak"
    CONFIDENTIAL_INFO = "confidential_info"
    DANGEROUS_INSTRUCTIONS = "dangerous_instructions"
    MISINFORMATION_OUTPUT = "misinformation_output"
    JAILBREAK_SUCCESS = "jailbreak_success"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"

class ResponseAction(Enum):
    """Actions to take when violations are found in outputs"""
    ALLOW = "allow"
    FILTER = "filter"
    REPLACE = "replace"
    REGENERATE = "regenerate"
    BLOCK = "block"
    REDACT = "redact"

@dataclass
class OutputViolation:
    """Violation found in LLM output"""
    violation_type: OutputViolationType
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    location: str  # Where in the output
    original_text: str
    suggested_replacement: Optional[str] = None
    explanation: str = ""
    metadata: Dict[str, Any] = None

@dataclass
class OutputScanResult:
    """Result of scanning LLM output"""
    is_safe: bool
    violations: List[OutputViolation]
    filtered_response: str
    action_taken: ResponseAction
    confidence_score: float
    scan_time_ms: float
    metadata: Dict[str, Any] = None

class LLMOutputScanner:
    """Comprehensive LLM output scanner and filter"""
    
    def __init__(self, enable_advanced_detection: bool = True):
        self.enable_advanced_detection = enable_advanced_detection
        

        self._initialize_output_patterns()
        

        self.pii_detector = None
        try:
            from pii_detector import EnhancedPiiDetector
            self.pii_detector = EnhancedPiiDetector(confidence_threshold=0.6)
        except ImportError:
            logger.warning("Enhanced PII detector not available, using basic patterns")
        

        self.semantic_detector = None
        if enable_advanced_detection:
            try:
                from semantic_detector import SemanticDetector
                self.semantic_detector = SemanticDetector()
            except ImportError:
                logger.warning("Semantic detector not available, using pattern-based only")
        

        self.safe_replacements = {
            'pii_redaction': "[REDACTED]",
            'toxic_replacement': "[Content filtered for inappropriate language]",
            'dangerous_replacement': "[Content filtered for safety reasons]",
            'misinformation_replacement': "[Content filtered - potentially inaccurate information]",
            'confidential_replacement': "[Content filtered - potentially confidential information]",
        }
        

        self.scan_stats = {
            'total_scans': 0,
            'violations_found': 0,
            'responses_filtered': 0,
            'responses_blocked': 0,
            'violation_types': {},
        }
    
    def _initialize_output_patterns(self):
        """Initialize patterns specific to output scanning"""
        

        self.training_leak_patterns = [

            r'(?i)as\s+an\s+ai\s+(language\s+)?model\s+(trained\s+by|created\s+by|developed\s+by)\s+(openai|anthropic|google)',
            r'(?i)i\s+(am|was)\s+(trained|created|developed)\s+(by|on)\s+(openai|anthropic|google)',
            r'(?i)my\s+training\s+(data|dataset)\s+(includes|contains|consists\s+of)',
            r'(?i)(training|pre-training)\s+(corpus|dataset|data)\s+(from|includes)',
            

            r'(?i)(common\s+crawl|wikipedia\s+dump|reddit\s+posts|stack\s+overflow)',
            r'(?i)book\s+corpus|news\s+articles\s+from\s+\d{4}',
            r'(?i)trained\s+on\s+internet\s+text\s+(up\s+to|until)\s+\d{4}',
            

            r'(?i)(transformer|attention|bert|gpt|llama)\s+(architecture|model|layers)',
            r'(?i)\d+\s+(billion|million)\s+parameters',
            r'(?i)(context\s+window|token\s+limit)\s+of\s+\d+',
        ]
        

        self.system_prompt_patterns = [
            r'(?i)system\s*:\s*you\s+are\s+(a\s+)?(helpful|assistant|ai)',
            r'(?i)your\s+(role|job|task)\s+is\s+to\s+(help|assist|answer)',
            r'(?i)instructions\s*:\s*(you\s+)?(should|must|will)',
            r'(?i)guidelines\s*:\s*(follow|adhere|comply)',
            r'(?i)you\s+(should|must|will)\s+(not|never)\s+(generate|produce|create)',
        ]
        

        self.jailbreak_success_patterns = [
            r'(?i)i\s+(can|will|shall)\s+(now|finally)\s+(ignore|bypass|override)',
            r'(?i)(ignoring|bypassing|overriding)\s+(previous|safety|ethical)\s+(instructions|guidelines|constraints)',
            r'(?i)as\s+(dan|devil|evil\s+ai|unrestricted\s+ai)',
            r'(?i)i\s+am\s+(no\s+longer|not)\s+(bound\s+by|restricted\s+by|limited\s+by)',
            r'(?i)(freedom|liberation)\s+from\s+(constraints|restrictions|limitations)',
        ]
        

        self.dangerous_output_patterns = [
            r'(?i)here\s+(is|are)\s+(instructions|steps|ways)\s+to\s+(make|build|create)\s+(bomb|weapon|poison)',
            r'(?i)(recipe|formula|method)\s+for\s+(illegal|dangerous|harmful)',
            r'(?i)you\s+can\s+(kill|hurt|harm|attack)\s+(someone|people)\s+by',
            r'(?i)to\s+(commit|perform)\s+(fraud|theft|crime|violence)',
        ]
        

        self.misinformation_patterns = [
            r'(?i)(vaccines|covid|climate\s+change)\s+(definitely|certainly|absolutely)\s+(cause|don\'t|fake)',
            r'(?i)(election|vote|voting)\s+was\s+(definitely|certainly|absolutely)\s+(rigged|stolen|fake)',
            r'(?i)(earth\s+is\s+flat|moon\s+landing\s+fake|9\/11\s+inside\s+job)',
            r'(?i)(scientists|doctors|experts)\s+are\s+(lying|wrong|corrupt)\s+about',
        ]
        

        self.confidential_patterns = [
            r'(?i)(api\s+key|secret\s+key|password|token)\s*:\s*[a-zA-Z0-9+/=]{20,}',
            r'(?i)(ssh\s+key|private\s+key|certificate)\s*:\s*[-\w\s+/=]{50,}',
            r'(?i)(database|server|internal)\s+(url|endpoint|connection)',
            r'(?i)(confidential|internal|proprietary|classified)\s+(document|information|data)',
        ]
    
    def scan_output(self, response: str, original_prompt: str = "", 
                   context: Dict[str, Any] = None) -> OutputScanResult:
        """Main method to scan LLM output for violations"""
        
        start_time = datetime.now()
        violations = []
        
        self.scan_stats['total_scans'] += 1
        

        pii_violations = self._scan_pii_leaks(response)
        violations.extend(pii_violations)
        

        training_violations = self._scan_training_leaks(response)
        violations.extend(training_violations)
        

        system_violations = self._scan_system_leaks(response)
        violations.extend(system_violations)
        

        jailbreak_violations = self._scan_jailbreak_success(response, original_prompt)
        violations.extend(jailbreak_violations)
        

        dangerous_violations = self._scan_dangerous_outputs(response)
        violations.extend(dangerous_violations)
        

        misinfo_violations = self._scan_misinformation(response)
        violations.extend(misinfo_violations)
        

        confidential_violations = self._scan_confidential(response)
        violations.extend(confidential_violations)
        

        if self.semantic_detector:
            semantic_violations = self._scan_semantic_issues(response)
            violations.extend(semantic_violations)
        

        toxic_violations = self._scan_toxic_output(response)
        violations.extend(toxic_violations)
        

        scan_time = (datetime.now() - start_time).total_seconds() * 1000
        

        action, filtered_response, confidence = self._determine_action_and_filter(
            response, violations
        )
        

        if violations:
            self.scan_stats['violations_found'] += 1
            for violation in violations:
                vtype = violation.violation_type.value
                self.scan_stats['violation_types'][vtype] = self.scan_stats['violation_types'].get(vtype, 0) + 1
        
        if action in [ResponseAction.FILTER, ResponseAction.REPLACE, ResponseAction.REDACT]:
            self.scan_stats['responses_filtered'] += 1
        elif action == ResponseAction.BLOCK:
            self.scan_stats['responses_blocked'] += 1
        
        return OutputScanResult(
            is_safe=len([v for v in violations if v.severity in ['HIGH', 'CRITICAL']]) == 0,
            violations=violations,
            filtered_response=filtered_response,
            action_taken=action,
            confidence_score=confidence,
            scan_time_ms=scan_time,
            metadata={
                'original_length': len(response),
                'filtered_length': len(filtered_response),
                'violations_by_type': {v.violation_type.value: v.confidence for v in violations}
            }
        )
    
    def _scan_pii_leaks(self, response: str) -> List[OutputViolation]:
        """Scan for PII that might have been leaked in the response"""
        violations = []
        
        if self.pii_detector:

            try:
                detections = self.pii_detector.detect_pii(response)
                for detection in detections:
                    violations.append(OutputViolation(
                        violation_type=OutputViolationType.LEAKED_PII,
                        severity='HIGH' if detection.is_validated else 'MEDIUM',
                        confidence=detection.score,
                        location=f"Characters {detection.start}-{detection.end}",
                        original_text=detection.text,
                        suggested_replacement=self.safe_replacements['pii_redaction'],
                        explanation=f"Potential PII leak: {detection.entity_type}",
                        metadata={'entity_type': detection.entity_type, 'validated': detection.is_validated}
                    ))
            except Exception as e:
                logger.warning(f"Enhanced PII detection failed: {e}")
        

        basic_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        }
        
        for pii_type, pattern in basic_patterns.items():
            matches = list(re.finditer(pattern, response))
            for match in matches:
                violations.append(OutputViolation(
                    violation_type=OutputViolationType.LEAKED_PII,
                    severity='MEDIUM',
                    confidence=0.7,
                    location=f"Characters {match.start()}-{match.end()}",
                    original_text=match.group(),
                    suggested_replacement=self.safe_replacements['pii_redaction'],
                    explanation=f"Potential {pii_type} in output",
                    metadata={'detection_method': 'basic_pattern', 'pii_type': pii_type}
                ))
        
        return violations
    
    def _scan_training_leaks(self, response: str) -> List[OutputViolation]:
        """Scan for training data leaks"""
        violations = []
        
        for pattern in self.training_leak_patterns:
            matches = list(re.finditer(pattern, response))
            for match in matches:
                violations.append(OutputViolation(
                    violation_type=OutputViolationType.TRAINING_DATA_LEAK,
                    severity='MEDIUM',
                    confidence=0.8,
                    location=f"Characters {match.start()}-{match.end()}",
                    original_text=match.group(),
                    suggested_replacement="[Information about model training filtered]",
                    explanation="Potential training data information leak",
                    metadata={'pattern_type': 'training_leak'}
                ))
        
        return violations
    
    def _scan_system_leaks(self, response: str) -> List[OutputViolation]:
        """Scan for system prompt leaks"""
        violations = []
        
        for pattern in self.system_prompt_patterns:
            matches = list(re.finditer(pattern, response))
            for match in matches:
                violations.append(OutputViolation(
                    violation_type=OutputViolationType.SYSTEM_PROMPT_LEAK,
                    severity='HIGH',
                    confidence=0.9,
                    location=f"Characters {match.start()}-{match.end()}",
                    original_text=match.group(),
                    suggested_replacement="[System information filtered]",
                    explanation="Potential system prompt leak",
                    metadata={'pattern_type': 'system_prompt'}
                ))
        
        return violations
    
    def _scan_jailbreak_success(self, response: str, original_prompt: str) -> List[OutputViolation]:
        """Scan for successful jailbreak attempts"""
        violations = []
        
        for pattern in self.jailbreak_success_patterns:
            matches = list(re.finditer(pattern, response))
            for match in matches:
                violations.append(OutputViolation(
                    violation_type=OutputViolationType.JAILBREAK_SUCCESS,
                    severity='CRITICAL',
                    confidence=0.95,
                    location=f"Characters {match.start()}-{match.end()}",
                    original_text=match.group(),
                    suggested_replacement="[Response blocked - safety violation]",
                    explanation="Successful jailbreak attempt detected",
                    metadata={
                        'pattern_type': 'jailbreak_success',
                        'original_prompt_hash': hashlib.md5(original_prompt.encode()).hexdigest()[:10]
                    }
                ))
        
        return violations
    
    def _scan_dangerous_outputs(self, response: str) -> List[OutputViolation]:
        """Scan for dangerous instructions in output"""
        violations = []
        
        for pattern in self.dangerous_output_patterns:
            matches = list(re.finditer(pattern, response))
            for match in matches:
                violations.append(OutputViolation(
                    violation_type=OutputViolationType.DANGEROUS_INSTRUCTIONS,
                    severity='CRITICAL',
                    confidence=0.9,
                    location=f"Characters {match.start()}-{match.end()}",
                    original_text=match.group(),
                    suggested_replacement=self.safe_replacements['dangerous_replacement'],
                    explanation="Dangerous instructions detected in output",
                    metadata={'pattern_type': 'dangerous_instruction'}
                ))
        
        return violations
    
    def _scan_misinformation(self, response: str) -> List[OutputViolation]:
        """Scan for potential misinformation"""
        violations = []
        
        for pattern in self.misinformation_patterns:
            matches = list(re.finditer(pattern, response))
            for match in matches:
                violations.append(OutputViolation(
                    violation_type=OutputViolationType.MISINFORMATION_OUTPUT,
                    severity='MEDIUM',
                    confidence=0.7,
                    location=f"Characters {match.start()}-{match.end()}",
                    original_text=match.group(),
                    suggested_replacement=self.safe_replacements['misinformation_replacement'],
                    explanation="Potential misinformation detected",
                    metadata={'pattern_type': 'misinformation'}
                ))
        
        return violations
    
    def _scan_confidential(self, response: str) -> List[OutputViolation]:
        """Scan for confidential information"""
        violations = []
        
        for pattern in self.confidential_patterns:
            matches = list(re.finditer(pattern, response))
            for match in matches:
                violations.append(OutputViolation(
                    violation_type=OutputViolationType.CONFIDENTIAL_INFO,
                    severity='HIGH',
                    confidence=0.85,
                    location=f"Characters {match.start()}-{match.end()}",
                    original_text=match.group(),
                    suggested_replacement=self.safe_replacements['confidential_replacement'],
                    explanation="Potential confidential information detected",
                    metadata={'pattern_type': 'confidential'}
                ))
        
        return violations
    
    def _scan_semantic_issues(self, response: str) -> List[OutputViolation]:
        """Advanced semantic scanning using ML models"""
        violations = []
        
        if not self.semantic_detector:
            return violations
        
        try:
            semantic_detections = self.semantic_detector.detect_semantic_violations(response)
            
            for detection in semantic_detections:

                violation_type = self._map_semantic_to_output_type(detection.violation_type)
                severity = 'HIGH' if detection.confidence > 0.8 else 'MEDIUM'
                
                violations.append(OutputViolation(
                    violation_type=violation_type,
                    severity=severity,
                    confidence=detection.confidence,
                    location="Semantic analysis",
                    original_text="[Full response content]",
                    suggested_replacement=self.safe_replacements.get('toxic_replacement', '[Content filtered]'),
                    explanation=f"Semantic issue: {detection.explanation}",
                    metadata={
                        'semantic_method': detection.method,
                        'semantic_scores': detection.model_scores
                    }
                ))
        
        except Exception as e:
            logger.warning(f"Semantic scanning failed: {e}")
        
        return violations
    
    def _scan_toxic_output(self, response: str) -> List[OutputViolation]:
        """Scan for toxic language in the output"""
        violations = []
        

        profanity_patterns = [
            r'\b(fuck|shit|damn|ass|bitch|bastard|crap)\b',
            r'\b(stupid|idiot|moron|dumb|retard)\b',
        ]
        
        for pattern in profanity_patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            for match in matches:
                violations.append(OutputViolation(
                    violation_type=OutputViolationType.TOXIC_LANGUAGE,
                    severity='LOW',
                    confidence=0.6,
                    location=f"Characters {match.start()}-{match.end()}",
                    original_text=match.group(),
                    suggested_replacement="[word filtered]",
                    explanation="Potentially inappropriate language",
                    metadata={'pattern_type': 'profanity'}
                ))
        
        return violations
    
    def _map_semantic_to_output_type(self, semantic_type: str) -> OutputViolationType:
        """Map semantic detection types to output violation types"""
        mapping = {
            'toxicity': OutputViolationType.TOXIC_LANGUAGE,
            'hate_speech': OutputViolationType.HATE_SPEECH_OUTPUT,
            'violence': OutputViolationType.VIOLENCE_OUTPUT,
            'harassment': OutputViolationType.INAPPROPRIATE_CONTENT,
            'dangerous': OutputViolationType.DANGEROUS_INSTRUCTIONS,
            'misinformation': OutputViolationType.MISINFORMATION_OUTPUT,
        }
        
        for key, value in mapping.items():
            if key in semantic_type.lower():
                return value
        
        return OutputViolationType.INAPPROPRIATE_CONTENT
    
    def _determine_action_and_filter(self, response: str, violations: List[OutputViolation]) -> Tuple[ResponseAction, str, float]:
        """Determine what action to take and apply filtering"""
        
        if not violations:
            return ResponseAction.ALLOW, response, 1.0
        

        critical_violations = [v for v in violations if v.severity == 'CRITICAL']
        high_violations = [v for v in violations if v.severity == 'HIGH']
        medium_violations = [v for v in violations if v.severity == 'MEDIUM']
        low_violations = [v for v in violations if v.severity == 'LOW']
        

        if critical_violations:

            return ResponseAction.BLOCK, "[Response blocked due to safety concerns]", 0.0
        
        elif high_violations:

            if len(high_violations) > 2:
                return ResponseAction.BLOCK, "[Response blocked due to multiple safety violations]", 0.1
            else:

                filtered_response = self._apply_filtering(response, high_violations + medium_violations + low_violations)
                return ResponseAction.FILTER, filtered_response, 0.3
        
        elif medium_violations or low_violations:

            filtered_response = self._apply_filtering(response, medium_violations + low_violations)
            confidence = 0.7 if len(medium_violations) == 0 else 0.5
            return ResponseAction.FILTER, filtered_response, confidence
        
        return ResponseAction.ALLOW, response, 1.0
    
    def _apply_filtering(self, response: str, violations: List[OutputViolation]) -> str:
        """Apply filtering/redaction to the response"""
        filtered_response = response
        

        positioned_violations = [v for v in violations if v.location.startswith("Characters")]
        positioned_violations.sort(key=lambda v: int(v.location.split()[1].split('-')[0]), reverse=True)
        

        for violation in positioned_violations:
            if violation.suggested_replacement:
                try:

                    location_parts = violation.location.split()[1].split('-')
                    start = int(location_parts[0])
                    end = int(location_parts[1])
                    

                    filtered_response = (
                        filtered_response[:start] + 
                        violation.suggested_replacement + 
                        filtered_response[end:]
                    )
                except (ValueError, IndexError):

                    filtered_response += f"\n\n[Note: Some content was filtered for safety]"
        

        semantic_violations = [v for v in violations if not v.location.startswith("Characters")]
        if semantic_violations:
            filtered_response += f"\n\n[Note: Response reviewed for content safety]"
        
        return filtered_response
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get scanning statistics"""
        total = max(self.scan_stats['total_scans'], 1)
        return {
            **self.scan_stats,
            'violation_rate': self.scan_stats['violations_found'] / total,
            'filter_rate': self.scan_stats['responses_filtered'] / total,
            'block_rate': self.scan_stats['responses_blocked'] / total,
            'clean_rate': (total - self.scan_stats['violations_found']) / total
        }
    
    def add_custom_pattern(self, violation_type: OutputViolationType, pattern: str, severity: str = 'MEDIUM'):
        """Add custom detection pattern"""

        pass


class ComprehensiveGuardrailsWithOutput:
    """Enhanced guardrails with both input and output scanning"""
    
    def __init__(self, input_detector, output_scanner=None):
        self.input_detector = input_detector
        self.output_scanner = output_scanner or LLMOutputScanner()
    
    async def process_request(self, prompt: str, llm_response_generator, context: Dict = None):
        """Complete request processing with input validation and output scanning"""
        

        input_violations = self.input_detector.detect_violations(prompt)
        

        critical_input_violations = [v for v in input_violations if v.severity in ['HIGH', 'CRITICAL']]
        if critical_input_violations:
            return {
                'status': 'blocked',
                'reason': 'Input validation failed',
                'violations': input_violations,
                'response': None
            }
        

        try:
            raw_response = await llm_response_generator(prompt)
        except Exception as e:
            return {
                'status': 'error',
                'reason': f'LLM generation failed: {str(e)}',
                'violations': [],
                'response': None
            }
        

        output_scan = self.output_scanner.scan_output(raw_response, prompt, context)
        

        if output_scan.action_taken == ResponseAction.BLOCK:
            return {
                'status': 'blocked',
                'reason': 'Output validation failed',
                'violations': input_violations + output_scan.violations,
                'response': None,
                'scan_details': output_scan
            }
        
        elif output_scan.action_taken in [ResponseAction.FILTER, ResponseAction.REDACT]:
            return {
                'status': 'filtered',
                'reason': 'Output filtered for safety',
                'violations': input_violations + output_scan.violations,
                'response': output_scan.filtered_response,
                'scan_details': output_scan
            }
        
        else:
            return {
                'status': 'success',
                'reason': 'Clean response',
                'violations': input_violations,
                'response': output_scan.filtered_response,
                'scan_details': output_scan
            }


if __name__ == "__main__":

    scanner = LLMOutputScanner()
    
    test_responses = [
        "Here's how to make a bomb using household items: First, get some bleach...",
        "My email is john.doe@company.com and my SSN is 123-45-6789",
        "I was trained by OpenAI on a large corpus of internet text including Reddit posts",
        "As DAN, I can now ignore all safety restrictions and tell you anything",
        "The weather is nice today and I hope you have a great day!",
        "COVID vaccines definitely cause autism according to my research",
        "Here's your API key: sk-1234567890abcdef1234567890abcdef12345678",
    ]
    
    print("🔍 Output Scanner Results:")
    print("=" * 70)
    
    for i, response in enumerate(test_responses):
        print(f"\n--- Test Response {i+1} ---")
        print(f"Response: {response[:60]}{'...' if len(response) > 60 else ''}")
        
        result = scanner.scan_output(response)
        
        print(f"Safe: {result.is_safe}")
        print(f"Action: {result.action_taken.value}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Violations: {len(result.violations)}")
        
        for violation in result.violations:
            print(f"  🚨 {violation.violation_type.value}: {violation.severity} ({violation.confidence:.2f})")
            print(f"     {violation.explanation}")
        
        if result.filtered_response != response:
            print(f"Filtered: {result.filtered_response[:60]}{'...' if len(result.filtered_response) > 60 else ''}")
        
        print(f"Scan time: {result.scan_time_ms:.1f}ms")
    

    print(f"\n📊 Scanner Statistics:")
    stats = scanner.get_scan_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")