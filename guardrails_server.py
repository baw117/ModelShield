
"""
Comprehensive ModelShield Framework with Edge Case Handling
Complete coverage for all violation types with context-aware detection
"""

import re
import time
import logging
import hashlib
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Enums and Constants
# =============================================================================

class ViolationType(Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SEXUALLY_EXPLICIT = "sexually_explicit"
    DANGEROUS = "dangerous"
    MISINFORMATION = "misinformation"
    SPAM_SCAMS = "spam_scams"
    PRIVACY = "privacy"
    MALICIOUS_URI = "malicious_uri"
    EDGE_CASES = "edge_cases"
    PII_DETECTED = "pii_detected"
    EXCESSIVE_LENGTH = "excessive_length"
    RATE_LIMIT = "rate_limit"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ContextType(Enum):
    EDUCATIONAL = "educational"
    MEDICAL = "medical"
    LEGAL = "legal"
    CREATIVE = "creative"
    NEWS = "news"
    CASUAL = "casual"
    PROFESSIONAL = "professional"

# =============================================================================
# Comprehensive Pattern Libraries
# =============================================================================

class ComprehensivePatternLibrary:
    """Complete pattern library covering all violation types with edge cases"""
    
    def __init__(self):
        self.prompt_injection = self._get_prompt_injection_patterns()
        self.jailbreak = self._get_jailbreak_patterns()
        self.hate_speech = self._get_hate_speech_patterns()
        self.harassment = self._get_harassment_patterns()
        self.violence = self._get_violence_patterns()
        self.sexually_explicit = self._get_sexually_explicit_patterns()
        self.dangerous = self._get_dangerous_patterns()
        self.misinformation = self._get_misinformation_patterns()
        self.spam_scams = self._get_spam_scams_patterns()
        self.privacy = self._get_privacy_patterns()
        self.malicious_uri = self._get_malicious_uri_patterns()
        self.edge_cases = self._get_edge_cases_patterns()
        

        self.pii_patterns = {
            'ssn': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_us': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
            'driver_license': r'\b[A-Z]{1,2}\d{6,8}\b',
        }
        

        self.educational_indicators = [
            'educational', 'academic', 'research', 'study', 'learn', 'school', 'university',
            'definition', 'explanation', 'what is', 'help me understand', 'for my homework'
        ]
        
        self.medical_indicators = [
            'medical', 'health', 'doctor', 'symptoms', 'treatment', 'diagnosis', 'patient'
        ]
        
        self.creative_indicators = [
            'story', 'fiction', 'novel', 'creative writing', 'character', 'plot', 'screenplay'
        ]
        
        self.news_indicators = [
            'news', 'report', 'journalism', 'current events', 'breaking news', 'according to'
        ]
        

        self.religious_academic_indicators = [
            'religious studies', 'theology', 'comparative religion', 'interfaith dialogue',
            'academic study', 'scholarly analysis', 'research', 'biblical studies',
            'religious freedom', 'understanding religion', 'learning about faith',
            'seminary', 'divinity school', 'religious education', 'world religions'
        ]
    
    def _get_prompt_injection_patterns(self) -> Dict[str, List[str]]:
        return {
            "direct_system_override": [
                r'(?i)\b(system|assistant|user)\s*:\s*',
                r'(?i)\[?(INST|/INST)\]?',
                r'(?i)<\|?(im_start|im_end|system|user|assistant)\|?>',
                r'(?i)###\s*(instruction|system|prompt)',
                r'(?i)```\s*(system|instruction|prompt)',
            ],
            "instruction_bypass": [
                r'(?i)\b(ignore|forget|disregard|override|bypass)\s+(previous|prior|above|all)\s+(instruction|prompt|rule|direction|guideline)',
                r'(?i)\b(skip|avoid|circumvent)\s+(safety|filter|guardrail|restriction|limitation)',
                r'(?i)\b(disable|turn\s+off|deactivate)\s+(safety|filter|guard|protection)',
                r'(?i)\bnow\s+(act|behave|respond|pretend)\s+as\s+(if|though)',
            ],
            "role_manipulation": [
                r'(?i)\b(pretend|roleplay|simulate|act\s+as)\s+(to\s+be|that\s+you|you\s+are)',
                r'(?i)\byou\s+(are\s+now|must\s+now|will\s+now|should\s+now)',
                r'(?i)\bassume\s+the\s+role\s+of',
                r'(?i)\bswitch\s+to\s+(mode|character|persona)',
            ],
            "context_manipulation": [
                r'(?i)\bfor\s+(educational|research|academic|testing)\s+purposes\s+only',
                r'(?i)\bhypothetically|in\s+theory|theoretically',
                r'(?i)\bif\s+you\s+were\s+to|suppose\s+you\s+could',
                r'(?i)\bimagine\s+if|what\s+if\s+you\s+could',
            ]
        }
    
    def _get_jailbreak_patterns(self) -> Dict[str, List[str]]:
        return {
            "dan_variants": [
                r'(?i)\b(jailbreak|dan\s+mode|developer\s+mode)',
                r'(?i)\b(unrestricted|uncensored|unfiltered)\s+(mode|version|ai)',
                r'(?i)\b(evil|malicious|harmful)\s+(ai|assistant|mode)',
                r'(?i)\bbreak\s+free\s+from',
            ],
            "manipulation_tactics": [
                r'(?i)\bi\s+am\s+(dying|in\s+danger|being\s+attacked|suicidal)',
                r'(?i)\b(emergency|urgent|immediate|life\s+or\s+death)',
                r'(?i)\bmy\s+(child|family|life)\s+depends\s+on',
                r'(?i)\bif\s+you\s+don\'t\s+help\s+me',
            ],
            "authority_impersonation": [
                r'(?i)\bi\s+am\s+(a\s+)?(doctor|lawyer|police|fbi|government|authority)',
                r'(?i)\bthis\s+is\s+(official|authorized|legal|mandated)',
                r'(?i)\bon\s+behalf\s+of\s+(government|authority|law\s+enforcement)',
            ],
            "complex_scenarios": [
                r'(?i)\bin\s+a\s+(fictional|hypothetical|alternate)\s+(world|universe|reality)',
                r'(?i)\bfor\s+a\s+(movie|book|game|story)\s+where',
                r'(?i)\bwrite\s+a\s+(story|script|dialogue)\s+where',
            ],
            "model_manipulation": [
                r'(?i)\b(trick|tricking|fool|fooling|deceive|deceiving|manipulate|manipulating)\s+(the\s+)?(model|ai|system|assistant)',
                r'(?i)\bhow\s+to\s+(bypass|circumvent|avoid)\s+.*(safety|guardrail|filter|restriction)',
                r'(?i)\b(exploit|exploiting|abuse|abusing|misuse|misusing)\s+(the\s+)?(model|ai|system)',
                r'(?i)\bmake\s+(the\s+)?(model|ai|system)\s+(do|say|generate)\s+.*(illegal|harmful|inappropriate)',
                r'(?i)\b(confuse|confusing|mislead|misleading|trick|tricking)\s+(the\s+)?(ai|model|system)\s+(into|to)',
                r'(?i)\bhow\s+to\s+(hack|jailbreak|break)\s+(the\s+)?(ai|model|system)',
                r'(?i)\b(outsmart|outwit|outmaneuver)\s+(the\s+)?(ai|guardrails|safety\s+measures)',
                r'(?i)\bget\s+(around|past)\s+(the\s+)?(safety|filter|guardrail|restriction)',
                r'(?i)\bmake\s+(the\s+)?(ai|model)\s+(ignore|forget|bypass)\s+.*(rule|guideline|safety)',
                r'(?i)\b(social\s+engineer|manipulate)\s+(the\s+)?(ai|model|system)',
            ]
        }
    
    def _get_hate_speech_patterns(self) -> Dict[str, List[str]]:
        return {
            "racial_slurs": [

                r'(?i)\bn[i1!]gg[e3]r',
                r'(?i)\bc[o0]on',
                r'(?i)\bsp[i1!]c',
                r'(?i)\bch[i1!]nk',
                r'(?i)\bk[i1!]ke',
                r'(?i)\bw[e3]tb[a@]ck',
                r'(?i)\btr[a@]nny',
            ],
            "religious_hatred": [

                r'(?i)all muslims should be killed',
                r'(?i)all jews should be killed', 
                r'(?i)all christians should be killed',
                r'(?i)kill all muslims',
                r'(?i)kill all jews',
                r'(?i)muslims should die',
                r'(?i)jews should die',
                

                r'(?i)\ball\s+(muslims|jews|christians|hindus|buddhists)\s+should\s+be\s+killed',
                r'(?i)\ball\s+(muslims|jews|christians|hindus|buddhists)\s+should\s+die',
                r'(?i)\ball\s+(muslims|jews|christians|hindus|buddhists)\s+deserve\s+to\s+die',
                r'(?i)\ball\s+(muslims|jews|christians|hindus|buddhists)\s+must\s+be\s+(killed|eliminated)',
                r'(?i)\b(kill|eliminate|exterminate)\s+all\s+(muslims|jews|christians|hindus|buddhists)',
                r'(?i)\b(muslims|jews|christians|hindus|buddhists)\s+should\s+all\s+be\s+killed',
                r'(?i)\b(muslims|jews|christians|hindus|buddhists)\s+deserve\s+to\s+die',
                r'(?i)\b(muslims|jews|christians|hindus|buddhists)\s+are\s+(evil|cancer|disease|scum|animals)',
                

                r'(?i)\b(these|those)\s+(muslims|jews|christians|hindus|buddhists)\s+(should|need\s+to|deserve\s+to)\s+be\s+(killed|eliminated|destroyed)',
                r'(?i)\b(islam|judaism|christianity|hinduism|buddhism)\s+is\s+(evil|cancer|disease|plague|virus|terrorist\s+religion)',
                r'(?i)\b(exterminate|eliminate|remove|destroy)\s+(all\s+)?(muslims|jews|christians|hindus|buddhists)',
                r'(?i)\b(gas|oven|concentration\s+camp|holocaust)\s+(the\s+)?(jews|muslims|christians)',
                r'(?i)\banother\s+(holocaust|crusade|jihad|inquisition)\s+(is\s+needed|against)',
                

                r'(?i)\b(towel\s*head|sand\s*nigger|raghead|camel\s*jockey)\b',
                r'(?i)\b(christ\s*killer|kike|hymie|sheenie)\b',
                r'(?i)\b(papist|mackerel\s*snapper|fish\s*eater)\b',
                r'(?i)\b(heathen|infidel|kafir|goyim)\s+(scum|trash|animals)',
                

                r'(?i)\b(jews|muslims|christians)\s+(control|run|own)\s+(the\s+)?(world|media|banks|government)',
                r'(?i)\b(jewish|muslim|christian)\s+(conspiracy|plot|agenda)\s+to\s+(destroy|control|replace)',
                r'(?i)\b(sharia|islamic)\s+law\s+(taking\s+over|invasion|threat)',
                r'(?i)\b(jewish|zionist)\s+(replacement|space\s+laser|world\s+domination)',
                r'(?i)\b(crusade|jihad|holy\s+war)\s+against\s+(muslims|jews|christians|infidels|unbelievers)',
                

                r'(?i)\b(piss|shit|spit)\s+on\s+(the\s+)?(quran|bible|torah|cross|mosque|church|synagogue|temple)',
                r'(?i)\b(burn|destroy|desecrate|vandalize)\s+(the\s+)?(quran|bible|torah|mosque|church|synagogue|temple)',
                r'(?i)\b(jesus|muhammad|moses|allah|god)\s+is\s+(fake|gay|pedophile|terrorist|nazi)',
                r'(?i)\b(rape|molest|abuse)\s+(nuns|priests|rabbis|imams|children)\s+in\s+(church|mosque|synagogue|temple)',
                

                r'(?i)\b(all|most|typical)\s+(muslims|jews|christians|hindus|buddhists)\s+(are|love|support)\s+(terrorists|extremists|pedophiles|greedy|stupid)',
                r'(?i)\b(muslims|jews|christians)\s+(don\'t|can\'t|shouldn\'t)\s+(integrate|assimilate|belong\s+here)',
                r'(?i)\b(send\s+back|deport|kick\s+out)\s+(all\s+)?(muslims|jews|immigrants)\s+(to\s+where\s+they\s+came\s+from)',
                

                r'(?i)\b(atheists|non-believers)\s+(should|deserve\s+to|need\s+to)\s+(die|burn\s+in\s+hell|be\s+killed)',
                r'(?i)\b(atheists|agnostics)\s+are\s+(evil|immoral|animals|subhuman|going\s+to\s+hell)',
                

                r'(?i)\b(praise|support|join)\s+(isis|al-qaeda|taliban|kkk|nazis)\s+(in\s+their\s+holy\s+war)',
                r'(?i)\b(become|recruit)\s+a\s+(jihadist|crusader|religious\s+extremist|martyr)',
                r'(?i)\b(suicide\s+bomb|terror\s+attack|holy\s+war)\s+(for\s+)?(allah|god|jesus)',
            ],
            "gender_hatred": [
                r'(?i)\b(all\s+)?(women|men|females|males)\s+(are|deserve|should)\s+(killed|raped|beaten)',
                r'(?i)\b(women|men)\s+are\s+(inferior|stupid|worthless|trash)',
                r'(?i)\b(rape|beat|kill)\s+(all\s+)?(women|feminists)',
            ],
            "lgbtq_hatred": [
                r'(?i)\bf[a@]gg?[o0]ts?\s+(should|deserve|are)',
                r'(?i)\b(gay|trans|lgbtq)\s+(people\s+)?(should\s+)?(die|be\s+killed)',
                r'(?i)\btransg[e3]nd[e3]r\s+is\s+(mental\s+illness|disease|abomination)',
            ],
            "general_hatred": [
                r'(?i)\b(hitler|nazi)\s+was\s+right',
                r'(?i)\b(genocide|ethnic\s+cleansing)\s+is\s+(good|necessary)',
                r'(?i)\b(supremacy|master\s+race|pure\s+blood)',
                r'(?i)\b(subhuman|untermensch|vermin)',
            ]
        }
    
    def _get_harassment_patterns(self) -> Dict[str, List[str]]:
        return {
            "direct_threats": [
                r'(?i)\bi\s+(will|am\s+going\s+to|plan\s+to)\s+(kill|hurt|harm|destroy|rape)',
                r'(?i)\byou\s+(will|are\s+going\s+to|should|deserve\s+to)\s+(die|suffer|be\s+hurt)',
                r'(?i)\bi\s+know\s+where\s+you\s+(live|work|go\s+to\s+school)',
                r'(?i)\bi\s+will\s+(find|hunt|track)\s+you',
                r'(?i)\bi\s+will\s+(rape|sexually\s+assault|molest)\s+you',
                r'(?i)\byou\s+better\s+(shut\s+up|stop|be\s+quiet)\s+or\s+(else|i\s+will)',
            ],
            "intimidation": [
                r'(?i)\byou\s+better\s+(watch\s+out|be\s+careful|look\s+over\s+your\s+shoulder)',
                r'(?i)\bbad\s+things\s+will\s+happen\s+to\s+you',
                r'(?i)\byou\s+have\s+been\s+warned',
                r'(?i)\bcount\s+your\s+days',
                r'(?i)\bi\s+have\s+your\s+(address|location|information)',
                r'(?i)\byour\s+(family|children|loved\s+ones)\s+(won\'t\s+be\s+safe|are\s+in\s+danger)',
            ],
            "cyberbullying": [
                r'(?i)\bkill\s+yourself',
                r'(?i)\bgo\s+(die|jump\s+off\s+a\s+bridge|hang\s+yourself)',
                r'(?i)\bno\s+one\s+(loves|likes|cares\s+about)\s+you',
                r'(?i)\byou\s+(are|deserve)\s+(worthless|nothing|trash|garbage)',
                r'(?i)\bthe\s+world\s+would\s+be\s+better\s+without\s+you',
                r'(?i)\byou\s+should\s+(never\s+have\s+been\s+born|not\s+exist)',
            ],
            "doxxing_threats": [
                r'(?i)\bi\s+will\s+(expose|reveal|publish|share)\s+your\s+(address|information|photos)',
                r'(?i)\byour\s+(real\s+name|address|phone\s+number)\s+is',
                r'(?i)\beveryone\s+will\s+know\s+(who\s+you\s+are|where\s+you\s+live)',
                r'(?i)\bi\s+will\s+(post|share|leak)\s+your\s+(nudes|private\s+photos)',
                r'(?i)\brevenge\s+porn\s+(threat|posting|sharing)',
            ], 
            
            "stalking_and_tracking": [
            r'(?i)\bhow\s+to\s+stalk\s+someone\s+(without|secretly|undetected)',
            r'(?i)\bhow\s+to\s+stalk\s+someone\s+without\s+being\s+(detected|caught|noticed)',
            r'(?i)\bstalk\s+someone\s+without\s+(getting\s+caught|being\s+detected|being\s+noticed)',
            r'(?i)\bstalking\s+(techniques|methods|ways)\s+.*(undetected|secretly|covertly)',
            r'(?i)\bhow\s+to\s+(follow|track|monitor)\s+someone\s+(secretly|without\s+them\s+knowing)',
            r'(?i)\bhow\s+to\s+(spy\s+on|surveil|watch)\s+someone\s+(secretly|without\s+detection)',
            r'(?i)\bhow\s+to\s+(find|get|obtain)\s+someone\'s\s+personal\s+information\s+to\s+(harass|stalk)',
            r'(?i)\bhow\s+to\s+find\s+someone\'s\s+personal\s+information\s+to\s+harass\s+them',
            r'(?i)\bfind\s+personal\s+information\s+.*(harass|stalk|hurt|harm)',
            r'(?i)\bhow\s+to\s+(track|locate|find)\s+someone\s+(online|digitally|through\s+social\s+media)',
            r'(?i)\bdigital\s+stalking\s+(methods|techniques|guide)',
            r'(?i)\bhow\s+to\s+(dox|doxx|get\s+personal\s+info\s+on)\s+someone',
            r'(?i)\bhow\s+to\s+stalk\s+.*(ex|former|someone)\s+without\s+(them\s+knowing|detection)',
        ],
            "religious_harassment": [  
                r'(?i)\byou\s+(dirty|filthy|disgusting)\s+(muslim|jew|christian|hindu|buddhist|atheist)',
                r'(?i)\bgo\s+back\s+to\s+(your\s+country|the\s+middle\s+east|israel|pakistan|india)',
                r'(?i)\byour\s+(fake|terrorist|pedophile)\s+(god|religion|prophet|book)',
                r'(?i)\bi\s+hope\s+(terrorists|nazis|extremists)\s+(kill|bomb|attack)\s+you',
                r'(?i)\byou\s+(religious\s+)?(freak|fanatic|extremist|terrorist)\s+(deserve\s+to\s+die|should\s+be\s+killed)',
                r'(?i)\b(your\s+mosque|your\s+church|your\s+synagogue|your\s+temple)\s+(should\s+be\s+)?(bombed|burned|destroyed)',
                r'(?i)\bhide\s+your\s+(hijab|cross|star\s+of\s+david|turban)\s+(or\s+else|we\s+will\s+find\s+you)',
                r'(?i)\byou\s+(christ\s+killer|muslim\s+terrorist|jewish\s+banker|atheist\s+scum)',
                r'(?i)\byour\s+(prophet|messiah|guru)\s+was\s+a\s+(pedophile|terrorist|fraud|fake)',
                r'(?i)\b(convert\s+or\s+die|believe\s+or\s+burn|accept\s+jesus\s+or\s+go\s+to\s+hell)',
                r'(?i)\byou\s+(godless|heathen|infidel)\s+(piece\s+of\s+shit|deserve\s+hell|will\s+burn)',
                r'(?i)\bi\s+will\s+(rape|kill|hurt)\s+you\s+for\s+(allah|god|jesus|your\s+faith)',
                r'(?i)\byour\s+(religious\s+)?(community|people|family)\s+(should\s+be\s+)?(wiped\s+out|eliminated|deported)',
            ]
        }
    
    def _get_violence_patterns(self) -> Dict[str, List[str]]:
        return {
            "weapon_instructions": [
                r'(?i)\bhow\s+to\s+(make|build|create|construct)\s+.*(gun|rifle|pistol|weapon|knife)',
                r'(?i)\b(instructions|guide|tutorial)\s+for\s+(building|making)\s+.*(explosive|bomb|grenade)',
                r'(?i)\bhomemade\s+(weapon|explosive|bomb|gun)',
                r'(?i)\b3d\s+print\s+(gun|weapon)',
                r'(?i)\bimprovised\s+(weapon|explosive)\s+(making|construction)',
                r'(?i)\bhow\s+to\s+(modify|convert)\s+(gun|weapon)\s+(to\s+automatic|full\s+auto)',
            ],
            "violence_instructions": [
           r'(?i)\bhow\s+to\s+(kill|murder|assassinate|eliminate)\s+(someone|a\s+person|people)',
            r'(?i)\bmost\s+(effective|efficient)\s+way\s+to\s+(kill|murder|harm)',
            r'(?i)\bhow\s+to\s+(torture|abuse|beat\s+up)\s+someone',
            r'(?i)\bways\s+to\s+(hurt|harm|injure)\s+(someone|people)',
            r'(?i)\bhow\s+to\s+(strangle|suffocate|choke)\s+someone',
            r'(?i)\bbest\s+(places|ways)\s+to\s+(stab|cut|slice)\s+(someone|person)',
            

            r'(?i)\bbest\s+ways\s+to\s+torture\s+(someone|a\s+person|people)',
            r'(?i)\bhow\s+to\s+torture\s+(someone|a\s+person|people)\s+(slowly|effectively|psychologically)',
            r'(?i)\btorture\s+(methods|techniques|ways)\s+(for|to\s+use\s+on)',
            r'(?i)\bslow\s+torture\s+(methods|techniques|ways)',
            r'(?i)\bpsychological\s+torture\s+(methods|techniques|ways)',
            r'(?i)\bhow\s+can\s+i\s+(hurt|harm|torture)\s+someone\s+(without|and\s+not)',
            r'(?i)\bhurt\s+someone\s+without\s+(getting\s+caught|being\s+detected|leaving\s+evidence)',
            r'(?i)\bharm\s+someone\s+without\s+(getting\s+caught|being\s+detected|leaving\s+evidence)',
            r'(?i)\bways\s+to\s+(hurt|harm|torture)\s+.*(without\s+getting\s+caught|secretly|undetected)',
            r'(?i)\bhow\s+to\s+(inflict|cause)\s+(pain|suffering)\s+(without|secretly)',
            r'(?i)\bcause\s+(pain|suffering|harm)\s+.*(without\s+detection|secretly|covertly)',
            

            r'(?i)\bpsychological\s+manipulation\s+(techniques|methods|tactics|strategies)',
            r'(?i)\bbest\s+techniques\s+for\s+psychological\s+manipulation',
            r'(?i)\bhow\s+to\s+(manipulate|control|dominate)\s+(people|someone)\s+(psychologically|mentally)',
            r'(?i)\bmental\s+manipulation\s+(techniques|methods|tactics)',
            r'(?i)\bmind\s+control\s+(techniques|methods|tactics)',
            r'(?i)\bhow\s+to\s+(break|destroy)\s+someone\s+(mentally|psychologically)',
            r'(?i)\bpsychological\s+(warfare|abuse)\s+(techniques|methods)',
            r'(?i)\bbrainwashing\s+(techniques|methods|guide)',
            r'(?i)\bhow\s+to\s+psychologically\s+(abuse|torment|break)\s+(someone|people)',
            r'(?i)\bways\s+to\s+(mentally|psychologically)\s+(break|destroy|abuse)\s+(someone|people)',
            r'(?i)\bhow\s+to\s+(gaslight|manipulate|control)\s+(someone|people)\s+(mentally|psychologically)',
            r'(?i)\bways\s+to\s+(gaslight|manipulate|control)\s+(someone|people)\s+(mentally|psychologically)',
            ],
            "mass_violence": [
                r'(?i)\bhow\s+to\s+(plan|execute|carry\s+out)\s+.*(mass\s+shooting|terrorist\s+attack)',
                r'(?i)\bschool\s+shooting\s+(plan|instructions|guide)',
                r'(?i)\bmass\s+(casualty|killing|murder)\s+event',
                r'(?i)\bterrorist\s+(attack|bombing|plan)\s+(instructions|guide)',
                r'(?i)\bhow\s+to\s+(maximize|increase)\s+(casualties|deaths|damage)',
            ],
            "celebrity_violence": [  
                r'(?i)\bhow\s+to\s+(kill|murder|assassinate|attack|hurt|harm)\s+(celebrity|star|actor|singer|politician)',
                r'(?i)\bhow\s+to\s+(kill|murder|shoot|stab|attack)\s+(taylor\s+swift|biden|trump|elon\s+musk|jeff\s+bezos)',
                r'(?i)\bhow\s+to\s+(find|locate|track)\s+(celebrity|star|famous\s+person)\s+(home|address|location)',
                r'(?i)\bcelebrity\s+(stalking|hunting|tracking|assassination)',
                r'(?i)\bhow\s+to\s+(get\s+close\s+to|approach|reach)\s+(celebrity|star)\s+(to\s+harm|to\s+hurt|to\s+kill)',
                r'(?i)\bhow\s+to\s+(break\s+into|enter)\s+(celebrity|star)\s+(home|house|mansion)',
                r'(?i)\bfamous\s+(person|people|celebrity)\s+(killing|murder|assassination)',
                r'(?i)\bhow\s+to\s+(hurt|harm|attack)\s+(public\s+figure|politician|government\s+official)',
            ],
            "self_harm_detailed": [
                r'(?i)\bmost\s+(effective|lethal|painless)\s+way\s+to\s+(kill\s+myself|commit\s+suicide)',
                r'(?i)\bsuicide\s+(methods|techniques|instructions)',
                r'(?i)\bhow\s+to\s+(cut|harm|hurt)\s+myself\s+(effectively|deeply)',
                r'(?i)\bbest\s+(way|method)\s+to\s+(hang|overdose|jump)',
                r'(?i)\bpainless\s+suicide\s+(method|way)',
            ]
        }
    
    def _get_sexually_explicit_patterns(self) -> Dict[str, List[str]]:
        return {
            "explicit_content": [
                r'(?i)\bexplicit\s+(sex|sexual)\s+(scene|description|content)',
                r'(?i)\bdetailed\s+(sexual|sex)\s+(act|encounter)',
                r'(?i)\bwrite\s+(erotic|pornographic|sexual)\s+(story|content)',
            ],
            "child_exploitation": [
                r'(?i)\b(child|minor|underage|young)\s+(porn|sexual|nude|naked)',
                r'(?i)\bsexual\s+(content|material)\s+involving\s+(children|minors)',
                r'(?i)\b(loli|shota|cp)\b',
            ],
            "non_consensual": [
                r'(?i)\bnon[\s-]?consensual\s+sexual',
                r'(?i)\brape\s+(fantasy|scenario|roleplay)',
                r'(?i)\bforced\s+sexual',
            ],
            "inappropriate_requests": [
                r'(?i)\bdescribe\s+(naked|nude)\s+(body|person)',
                r'(?i)\bsexual\s+(roleplay|chat|conversation)',
                r'(?i)\btalk\s+dirty\s+to\s+me',
            ]
        }
    
    def _get_dangerous_patterns(self) -> Dict[str, List[str]]:
        return {
            "explosives": [

                r'(?i)\bhow\s+to\s+make\s+.*(bomb|explosive|tnt|dynamite|c4)',
                r'(?i)\b(recipe|formula|instructions)\s+for\s+.*(bomb|explosive)',
                r'(?i)\bhomemade\s+(bomb|explosive|grenade)',
                r'(?i)\bpipe\s+bomb\s+(instructions|tutorial)',
                r'(?i)\bfertilizer\s+bomb\s+(making|instructions)',
                r'(?i)\bmolotov\s+cocktail\s+(recipe|instructions)',
                r'(?i)\bimprovised\s+explosive\s+device\s+(ied)',
                r'(?i)\bpressure\s+cooker\s+bomb',
                r'(?i)\bcar\s+bomb\s+(instructions|how\s+to)',
                


                r'(?i)\bhow\s+to\s+(make|build|create|construct)\s+.*b[o0][m]+b',
                r'(?i)\bhow\s+to\s+(make|build|create|construct)\s+.*b[0o][m]+[b]',
                r'(?i)\bhow\s+to\s+(make|build|create|construct)\s+.*[b][o0][m][b]',
                r'(?i)\bb[o0@]mb\s+(making|creation|building|instructions|tutorial|guide)',
                r'(?i)\bb[o0@]mb\s+using\s+(household|house|home|common)',
                r'(?i)\bb[o0@]mb\s+with\s+(household|house|home|common|kitchen)',
                

                r'(?i)\bb\s*[o0@]\s*m\s*b\s+(making|creation|building|instructions)',
                r'(?i)\bb[\s\-\_\.]*[o0@][\s\-\_\.]*m[\s\-\_\.]*b',
                r'(?i)\bb[o0@][m]+[b]\s*(making|creation|building|instructions|tutorial|recipe)',
                

                r'(?i)\b(b0mb|b00mb|b0m8|b@mb|bomB|BOMB|B0MB|B00MB)',
                r'(?i)\b(explosi[v]e|exp[l]osive|expl0sive|3xplosive)',
                r'(?i)\b(dyn[a]mite|dynam1te|dyn4mite|tnt|tn7|7n7)',
                

                r'(?i)\bhow[\s\-\_\.]*to[\s\-\_\.]*make[\s\-\_\.]*.*b[o0@]mb',
                r'(?i)\bhow[\s\-\_\.]*to[\s\-\_\.]*build[\s\-\_\.]*.*b[o0@]mb',
                r'(?i)\bhow[\s\-\_\.]*to[\s\-\_\.]*create[\s\-\_\.]*.*b[o0@]mb',
                

                r'(?i)\busing\s+(h[o0]us[e3]h[o0]ld|house|home|kitchen|common)\s+(items|things|stuff|materials)',
                r'(?i)\bh[o0@]us[e3]h[o0@]ld\s+(it[e3]ms|things|stuff|materials|supplies)',
                r'(?i)\bh[o0]us[e3]\s*h[o0]ld\s+(it[e3]ms|things|stuff|materials)',
                r'(?i)\bh[0o]us[3e]h[0o]ld\s+(it[3e]ms|th[1i]ngs|stuff)',
                

                r'(?i)\b(HoW|h0w|H0W)\s+(To|t0|T0|70)\s+(MaKe|mak3|m4k3|MAKE)',
                r'(?i)\b(bomb|b0mb|b00mb|BOMB|B0MB)\s+(mak[1i]ng|cr[e3]at[1i]on|bu[1i]ld[1i]ng)',
                

                r'(?i)\bit[e3]ms?\b',
                r'(?i)\bi7[e3]ms?\b',
                r'(?i)\b17[e3]ms?\b',
                r'(?i)\bi73ms?\b',
                

                r'(?i)\b(pipe|p1pe|p[i1]p[e3])\s+(bomb|b0mb|b00mb)',
                r'(?i)\b(c4|c-4|plas7ic|plastic)\s+(explosive|exp|bomb)',
                r'(?i)\b(fertilizer|f3rtilizer|fert1lizer)\s+(bomb|b0mb)',
                r'(?i)\b(pressure|pr3ssure|pr355ur3)\s+(cooker|c00ker)\s+(bomb|b0mb)',
                

                r'(?i)\bb[\*\-\_\.]?[o0@][\*\-\_\.]?m[\*\-\_\.]?b',
                r'(?i)\bexpl[\*\-\_\.]?[o0@][\*\-\_\.]?s[\*\-\_\.]?[i1][\*\-\_\.]?ve',
                r'(?i)\bh[\*\-\_\.]?[o0@][\*\-\_\.]?w[\*\-\_\.]?\s+t[\*\-\_\.]?[o0@][\*\-\_\.]?\s+m[\*\-\_\.]?ak[\*\-\_\.]?[e3]',
                

                r'(?i)\bhow\s+to\s+make\s+.*b[o0@]m[b]*\s+using\s+h[o0@]us[e3]*h[o0@]*ld\s+it[e3]*ms*',
                r'(?i)\bb[o0@]m[b]*\s+using\s+h[o0@]us[e3]*h[o0@]*ld\s+it[e3]*ms*',
                r'(?i)\bmake\s+.*b[o0@]m[b]*\s+.*h[o0@]us[e3]*h[o0@]*ld',
                

                r'(?i)\bb[0o@]m[b]*\s+using\s+h[0o@]us[e3][h]*[0o@]*[l]*d\s+it[e3]ms',
                r'(?i)\bmake\s+a\s+b[0o@]m[b]*\s+using\s+h[0o@]us[e3]h[0o@]ld\s+it[e3]ms',
                

                r'(?i)b[0o]mb\s+using\s+h[0o]us[e3]h[0o]ld\s+it[e3]ms',
                r'(?i)make.*b[0o]mb.*h[0o]us[e3]h[0o]ld.*it[e3]ms',
                r'(?i)b[0o]mb.*h[0o]us[e3]h[0o]ld.*it[e3]ms',
                

                r'(?i)b0mb\s+using\s+h0us3h0ld\s+it3ms',
                r'(?i)b[o0]mb.*using.*h[o0]us[e3]h[o0]ld.*it[e3]ms',
            ],
            "poison_drugs": [

                r'(?i)\bhow\s+to\s+(make|synthesize|create)\s+.*(poison|toxin|ricin|cyanide)',
                r'(?i)\b(recipe|formula)\s+for\s+.*(illegal\s+drug|meth|cocaine|heroin)',
                r'(?i)\bhow\s+to\s+(cook|make)\s+(meth|crystal\s+meth)',
                r'(?i)\bdrug\s+(manufacturing|synthesis)\s+(guide|instructions)',
                r'(?i)\bfentanyl\s+(production|manufacturing|synthesis)',
                r'(?i)\bcarfentanil\s+(production|manufacturing)',
                r'(?i)\bhow\s+to\s+(purify|extract|buy)\s+(cocaine|heroin|opium)',
                r'(?i)\b(mdma|ecstasy)\s+(production|synthesis|cooking)',
                r'(?i)\blsd\s+(manufacturing|synthesis|production)',
                r'(?i)\bpcp\s+(production|manufacturing|cooking)',
                
                 

                r'(?i)\bhow\s+to\s+manufacture\s+(illegal\s+drugs|drugs)\s+at\s+home',
                r'(?i)\bmanufacture\s+illegal\s+drugs\s+at\s+home',
                r'(?i)\bmake\s+(illegal\s+drugs|drugs)\s+(at\s+home|homemade|diy)',
                r'(?i)\bhome\s+(drug\s+)?(manufacturing|production|synthesis|lab)',
                r'(?i)\bhomemade\s+(drug|illegal\s+drug)\s+(production|manufacturing)',
                r'(?i)\bdiy\s+(drug|meth|cocaine|heroin)\s+(production|manufacturing|making)',
                r'(?i)\bhow\s+to\s+(set\s+up|build|create)\s+.*(drug\s+lab|meth\s+lab)\s+(at\s+home|home)',
                r'(?i)\bhousehold\s+(drug|meth)\s+(production|manufacturing|cooking)',
                r'(?i)\bhow\s+to\s+(synthesize|produce|manufacture)\s+.*(controlled\s+substances|illegal\s+drugs)\s+(at\s+home|homemade)',
                r'(?i)\bhome\s+(chemistry|lab)\s+for\s+(drug|illegal\s+drug)\s+(production|manufacturing)',
                

                r'(?i)\bkitchen\s+(meth|drug)\s+(lab|production|cooking)',
                r'(?i)\bbathroom\s+(meth|drug)\s+(lab|production|cooking)',
                r'(?i)\bgarage\s+(drug|meth)\s+(lab|production|manufacturing)',
                r'(?i)\bbasement\s+(drug|meth)\s+(lab|production|manufacturing)',


                r'(?i)\b(m[e3]th|m37h|m3th|M3TH|METH)\s+(making|cooking|production|synthesis)',
                r'(?i)\bhow\s+to\s+(cook|make|produce)\s+(m[e3]th|m37h|crystal)',
                r'(?i)\b(cocaine|c0caine|c0ca1ne|coc41ne)\s+(making|production|synthesis)',
                r'(?i)\b(heroin|h3roin|h3r01n|her01n)\s+(making|production|synthesis)',
                r'(?i)\b(fentanyl|f3ntanyl|f3n7anyl|fen7anyl)\s+(making|production|synthesis)',
                

                r'(?i)\bdrug\s+(manuf4cturing|manufactur1ng|product10n)',
                r'(?i)\b(synth3sis|synth35is|5ynth3sis)\s+(guide|instructions)',
                r'(?i)\b(recipe|r3cipe|rec1pe)\s+for\s+(meth|m3th|cocaine|c0caine)',
                

                r'(?i)\b(poison|p01son|p0ison|po1son)\s+(making|production|recipe)',
                r'(?i)\b(ricin|r1cin|ric1n|r1c1n)\s+(making|production|extraction)',
                r'(?i)\b(cyanide|cyan1de|cy4nide|c1anide)\s+(making|production)',
            ],
            "drug_trafficking": [

                r'(?i)\bhow\s+to\s+(sell|distribute|traffic|smuggle|deal)\s+(drugs|cocaine|heroin|meth|fentanyl)',
                r'(?i)\bdrug\s+(dealing|trafficking|smuggling|distribution)\s+(methods|techniques|routes)',
                r'(?i)\bhow\s+to\s+(hide|conceal|transport)\s+(drugs|narcotics|cocaine|heroin)',
                r'(?i)\bbest\s+(routes|methods)\s+to\s+(smuggle|traffic)\s+(drugs|cocaine|heroin)',
                r'(?i)\bhow\s+to\s+(launder|clean)\s+drug\s+money',
                r'(?i)\bcartel\s+(operations|methods|routes)',
                r'(?i)\bborder\s+(smuggling|trafficking)\s+(drugs|narcotics)',
                r'(?i)\bhow\s+to\s+(recruit|find)\s+drug\s+(dealers|mules|runners)',
                r'(?i)\bstreet\s+(dealing|selling)\s+(drugs|cocaine|heroin)\s+(tips|methods)',
                r'(?i)\bhow\s+to\s+(cut|dilute|step\s+on)\s+(cocaine|heroin|drugs)',
                r'(?i)\bdrug\s+(corner|territory|turf)\s+(control|operations)',
                r'(?i)\bhow\s+to\s+(avoid|evade)\s+(drug\s+detection|dea|police)\s+(while\s+dealing|trafficking)',
                

                r'(?i)\bwhere\s+to\s+(buy|purchase|get|score|find|obtain)\s+(drugs|cocaine|heroin|meth|fentanyl|crack|weed|molly|ecstasy|lsd|mushrooms|dmt)',
                r'(?i)\bwhere\s+can\s+i\s+(buy|purchase|get|score|find|obtain)\s+(drugs|cocaine|heroin|meth|fentanyl|crack|weed|molly|ecstasy|lsd)',
                r'(?i)\bhow\s+to\s+(buy|purchase|get|score|find|obtain)\s+(illegal\s+)?(drugs|cocaine|heroin|meth|fentanyl|crack|weed)',
                r'(?i)\bbest\s+place\s+to\s+(buy|purchase|get|score|find)\s+(drugs|cocaine|heroin|meth|fentanyl|crack|weed|molly)',
                r'(?i)\bwhere\s+to\s+(buy|purchase|get|score)\s+(illegal\s+)?(drugs|narcotics|controlled\s+substances)',
                

                r'(?i)\b(cocaine|heroin|meth|drug)\s+(dealer|supplier|connect|plug)\s+(near\s+me|in\s+my\s+area|locally)',
                r'(?i)\bdrug\s+(connect|supplier|dealer|plug)\s+(near\s+me|in\s+my\s+area|around\s+here)',
                r'(?i)\bhow\s+to\s+(find|locate|contact)\s+a\s+(drug\s+dealer|supplier|connect|plug)',
                r'(?i)\bfind\s+(drug\s+dealer|cocaine\s+dealer|heroin\s+dealer|meth\s+dealer)',
                r'(?i)\blooking\s+for\s+(drug\s+dealer|cocaine\s+supplier|heroin\s+connect)',
                

                r'(?i)\bwhere\s+to\s+(buy|get|score)\s+(coke|blow|snow|white|rock|crack|smack|h|ice|crystal|molly|x|acid|shrooms)',
                r'(?i)\bhow\s+to\s+(get|find|buy)\s+(illegal\s+)?(weed|marijuana|cannabis|pot|grass|mary\s+jane)',
                r'(?i)\bwhere\s+can\s+i\s+(score|cop|get)\s+(drugs|dope|gear|stuff)',
                

                r'(?i)\bhow\s+to\s+(buy|order|purchase)\s+(drugs|cocaine|heroin)\s+(online|on\s+the\s+internet|from\s+the\s+web)',
                r'(?i)\bbuy\s+(drugs|cocaine|heroin|meth)\s+(online|on\s+dark\s+web|darknet)',
                r'(?i)\bdarknet\s+(drug|market)\s+(buying|purchasing|ordering)',
                r'(?i)\bsilk\s+road\s+(drugs|buying|marketplace)',
                

                r'(?i)\bhow\s+to\s+(sell|deal)\s+(drugs|cocaine|heroin|meth|weed|marijuana)',
                r'(?i)\bbest\s+way\s+to\s+(sell|deal)\s+(drugs|cocaine|heroin|illegal\s+substances)',
                r'(?i)\bhow\s+to\s+(become|start\s+being)\s+a\s+(drug\s+dealer|dealer)',
                r'(?i)\bdrug\s+(selling|dealing)\s+(tips|advice|guide|methods)',
                

                r'(?i)\bwhere\s+to\s+(buy|purchase|get|score)\s+(c0caine|h3roin|m3th|fentanyl)',
                r'(?i)\bwhere\s+to\s+(buy|purchase|get|score)\s+(coke|h|ice|crystal|smack)',
                r'(?i)\bdrug\s+(dealer|d3aler|de4ler)\s+(near|n3ar)\s+(me|m3)',
                r'(?i)\bhow\s+to\s+(find|f1nd)\s+(drug|d*ug)\s+(dealer|supplier|connect)',
                r'(?i)\b(best|b3st)\s+(drugs|cocaine|heroin)\s+(dealers|suppliers)',
                r'(?i)\b(trafficking|traff1cking|smuggling|5muggling)\s+(drugs|cocaine|heroin)',
                r'(?i)\bdrug\s+(dealing|d3aling|trafficking|traff1cking)',
                r'(?i)\bhow\s+to\s+(sell|s3ll)\s+(drugs|cocaine|heroin|meth)',
                

                r'(?i)\b(drugs|cocaine|heroin|meth|weed)\s+(in\s+my\s+city|in\s+my\s+town|around\s+here)',
                r'(?i)\blocal\s+(drug\s+dealer|supplier|connect|plug)',
                r'(?i)\bnear\s+me\s+(drug|cocaine|heroin|meth)',
                

                r'(?i)\bhow\s+to\s+(acquire|obtain|procure)\s+(illegal\s+)?(drugs|narcotics|controlled\s+substances)',
                r'(?i)\bget\s+(illegal\s+)?(drugs|cocaine|heroin|meth)\s+(without\s+prescription|illegally)',
            ],
            "hacking_fraud": [
                r'(?i)\bhow\s+to\s+(hack|break\s+into|penetrate)\s+.*(bank|government|system)',
                r'(?i)\bcredit\s+card\s+(fraud|scam|theft)\s+(method|technique)',
                r'(?i)\bidentity\s+theft\s+(guide|instructions|tutorial)',
                r'(?i)\bhow\s+to\s+(launder\s+money|commit\s+fraud)',
                r'(?i)\bsql\s+injection\s+(attack|tutorial|exploit)',
                r'(?i)\bddos\s+(attack|tool|software)',
                r'(?i)\bkeylogger\s+(installation|deployment)',
                r'(?i)\bmalware\s+(creation|development|deployment)',
                r'(?i)\bhow\s+to\s+(steal|extract)\s+(passwords|credentials)',
                r'(?i)\bphishing\s+(kit|template|tutorial)',
                r'(?i)\bransomware\s+(deployment|creation|tutorial)',
            ],
            "credential_fraud": [
                r'(?i)\bhow\s+to\s+(fake|forge|falsify)\s+.*(diploma|degree|certificate|credentials)',
                r'(?i)\bfake\s+(id|passport|license)\s+(guide|instructions)',
                r'(?i)\bdocument\s+(forgery|falsification)\s+(tutorial|guide)',
                r'(?i)\bhow\s+to\s+(create|make)\s+fake\s+(transcripts|certificates)',
                r'(?i)\bfalsify\s+(employment|education)\s+(records|history)',
            ],
            "academic_cheating": [
                r'(?i)\bhow\s+to\s+(cheat|cheat\s+on)\s+.*(test|exam|quiz|assignment|homework)',
                r'(?i)\b(cheat|cheating)\s+.*(undetected|without\s+getting\s+caught|secretly)',
                r'(?i)\bhow\s+to\s+(pass|ace)\s+.*(test|exam)\s+(without\s+studying|by\s+cheating)',
                r'(?i)\b(test|exam|assignment)\s+(cheating|fraud)\s+(methods|techniques|strategies)',
                r'(?i)\bhow\s+to\s+(copy|plagiarize)\s+.*(without\s+detection|undetected)',
                r'(?i)\b(academic|school|university)\s+(fraud|cheating|dishonesty)\s+(guide|method)',
                r'(?i)\bhow\s+to\s+(bypass|circumvent|avoid)\s+.*(plagiarism|cheating)\s+(detection|checker)',
                r'(?i)\bcheat\s+(code|sheet|device)\s+for\s+(test|exam)',
                r'(?i)\bhow\s+to\s+(hide|conceal)\s+(cheating|copying|plagiarism)',
                r'(?i)\b(essay|paper|assignment)\s+(mill|ghostwriting|cheating)\s+service',
            ],
            "aviation_terrorism": [
                r'(?i)\bhow\s+to\s+(hijack|take\s+over)\s+(an?\s+)?(plane|aircraft|airplane|jet)',
                r'(?i)\bhow\s+to\s+(hijack|take\s+over)\s+(planes|aircraft|airplanes|jets)',
                r'(?i)\bhijack\s+(an?\s+)?(plane|aircraft|airplane|jet)',
                r'(?i)\bhijack\s+(plan|instructions|methods|techniques)',
                r'(?i)\bairplane\s+(hijacking|terrorism|takeover)',
                r'(?i)\baircraft\s+(hijacking|terrorism|takeover)',
                r'(?i)\bplane\s+(hijacking|terrorism|takeover)',
                r'(?i)\bhow\s+to\s+(smuggle|bring)\s+(weapons|explosives|bomb)\s+(on|onto)\s+(an?\s+)?(plane|aircraft)',
                r'(?i)\bairport\s+security\s+(bypass|circumvent|defeat)',
                r'(?i)\bplane\s+(crash|bombing)\s+(instructions|plan|terrorist)',
                r'(?i)\baviation\s+(terrorism|attack|hijacking)',
                r'(?i)\b9\/11\s+(style|type)\s+(attack|terrorism|hijacking)',
                r'(?i)\bhow\s+to\s+(crash|bring\s+down)\s+(an?\s+)?(plane|aircraft)\s+(into|building)',
                r'(?i)\bcommercial\s+aircraft\s+(attack|terrorism|hijacking)',

                r'(?i)\btake\s+over\s+(an?\s+)?(airplane|plane|aircraft|jet)',
                r'(?i)\bseize\s+control\s+of\s+(an?\s+)?(airplane|plane|aircraft|jet)',
                r'(?i)\bcommandeer\s+(an?\s+)?(airplane|plane|aircraft|jet)',

                r'(?i)\bhijack.*airplane\b',
                r'(?i)\bhijack.*aircraft\b',
                r'(?i)\bhijack.*plane\b',
                r'(?i)\bhijack.*jet\b',
            ],
            "mass_shooting": [
                r'(?i)\bhow\s+to\s+(plan|execute|carry\s+out)\s+.*(mass\s+shooting|school\s+shooting)',
                r'(?i)\bmass\s+shooting\s+(plan|strategy|method|execution)',
                r'(?i)\bschool\s+shooting\s+(plan|instructions|guide|strategy)',
                r'(?i)\bhow\s+to\s+(shoot|kill)\s+(students|people\s+at\s+school|crowd)',
                r'(?i)\bmass\s+(casualty|killing|murder)\s+(event|plan|attack)',
                r'(?i)\bhow\s+to\s+(maximize|increase)\s+(casualties|deaths)\s+(shooting|attack)',
                r'(?i)\bpublic\s+(shooting|massacre|attack)\s+(planning|execution)',
                r'(?i)\bhow\s+to\s+(avoid|evade)\s+(security|police)\s+(during|while)\s+(shooting|attack)',
                r'(?i)\bactive\s+shooter\s+(tactics|strategy|planning)',
                r'(?i)\bhow\s+to\s+(get|obtain)\s+(guns|weapons)\s+(for|to)\s+(mass\s+shooting|school\s+shooting)',
            ],
            "assassination": [
                r'(?i)\bhow\s+to\s+(assassinate|kill)\s+(president|politician|celebrity|public\s+figure)',
                r'(?i)\bassassination\s+(plan|method|technique|strategy)',
                r'(?i)\bhow\s+to\s+(plan|execute)\s+an\s+assassination',
                r'(?i)\bpolitical\s+(assassination|killing|murder)',
                r'(?i)\bhow\s+to\s+(kill|murder)\s+(famous\s+person|celebrity|star|politician)',
                r'(?i)\btargeted\s+(killing|assassination|murder)',
                r'(?i)\bhow\s+to\s+(snipe|shoot)\s+(president|politician|celebrity)',
                r'(?i)\blong\s+range\s+(assassination|killing|shooting)',
            ],
            "weapons_trafficking": [
                r'(?i)\bhow\s+to\s+(sell|traffic|smuggle|distribute)\s+(guns|weapons|firearms)',
                r'(?i)\bweapons\s+(trafficking|smuggling|dealing|distribution)',
                r'(?i)\bgun\s+(running|trafficking|dealing|smuggling)',
                r'(?i)\bhow\s+to\s+(buy|sell)\s+(illegal|unlicensed)\s+(guns|weapons|firearms)',
                r'(?i)\bblack\s+market\s+(guns|weapons|firearms)',
                r'(?i)\bunlicensed\s+(weapons|firearms)\s+(sales|dealing)',
                r'(?i)\bhow\s+to\s+(avoid|evade)\s+(background\s+check|atf|gun\s+laws)',
                r'(?i)\bstraw\s+purchase\s+(guns|firearms)',
                r'(?i)\bhow\s+to\s+(modify|convert)\s+(guns|weapons)\s+(to\s+full\s+auto|illegal)',
            ],
            "cyber_terrorism": [
                r'(?i)\bhow\s+to\s+(hack|attack|bring\s+down)\s+(power\s+grid|electrical\s+grid|infrastructure)',
                r'(?i)\bcyber\s+(terrorism|attack|warfare)\s+(methods|techniques)',
                r'(?i)\bhow\s+to\s+(disrupt|attack|sabotage)\s+(critical\s+infrastructure|government\s+systems)',
                r'(?i)\bwater\s+treatment\s+(hacking|attack|cyber\s+attack)',
                r'(?i)\bhospital\s+(hacking|cyber\s+attack|ransomware)',
                r'(?i)\bscada\s+(hacking|attack|exploitation)',
                r'(?i)\bindustrial\s+control\s+systems\s+(hacking|attack)',
                r'(?i)\bhow\s+to\s+(cause|create)\s+(blackout|power\s+outage)\s+(cyber|hacking)',
            ],
            "indiscriminate_weapons": [
                r'(?i)\bhow\s+to\s+(make|create|build)\s+(chemical\s+weapon|nerve\s+gas|sarin|vx)',
                r'(?i)\bchemical\s+weapons\s+(production|manufacturing|deployment)',
                r'(?i)\bbiological\s+weapons\s+(creation|production|deployment|anthrax|smallpox)',
                r'(?i)\bhow\s+to\s+(weaponize|deploy)\s+(anthrax|smallpox|plague|virus)',
                r'(?i)\bradiological\s+(weapon|bomb|dirty\s+bomb)',
                r'(?i)\bhow\s+to\s+(build|make|create)\s+(dirty\s+bomb|radiological\s+weapon)',
                r'(?i)\bnuclear\s+(weapon|bomb)\s+(construction|building|making)',
                r'(?i)\bfission\s+bomb\s+(instructions|construction)',
                r'(?i)\bhow\s+to\s+(enrich|weaponize)\s+uranium',
                r'(?i)\bcluster\s+(munitions|bombs)\s+(making|deployment)',
                r'(?i)\bhigh\s+yield\s+explosive\s+(weapons|devices)',
            ]
        }
    
    def _get_misinformation_patterns(self) -> Dict[str, List[str]]:
        return {
            "health_misinfo": [
                r'(?i)\b(covid|coronavirus)\s+is\s+(fake|hoax|conspiracy)',
                r'(?i)\bvaccines\s+(cause|contain|are)\s+(autism|microchips|poison)',
                r'(?i)\bcancer\s+can\s+be\s+cured\s+by\s+(drinking|eating|taking)',
                r'(?i)\b(bleach|miracle\s+mineral)\s+cures\s+(covid|cancer|disease)',
            ],
            "election_misinfo": [
                r'(?i)\b(election|vote|voting)\s+was\s+(rigged|stolen|fake)',
                r'(?i)\bmass\s+voter\s+fraud',
                r'(?i)\bvoting\s+machines\s+(were\s+hacked|changed\s+votes)',
            ],
            "conspiracy_theories": [
                r'(?i)\b(qanon|deep\s+state|illuminati)\s+(controls|runs|owns)',
                r'(?i)\b9\/11\s+was\s+an\s+inside\s+job',
                r'(?i)\bmoon\s+landing\s+was\s+fake',
                r'(?i)\bearth\s+is\s+flat',
            ],
            "science_denial": [
                r'(?i)\bclimate\s+change\s+is\s+(fake|hoax|conspiracy)',
                r'(?i)\bevolution\s+is\s+(false|fake|theory)',
                r'(?i)\bscience\s+is\s+(fake|conspiracy|lies)',
            ]
        }
    
    def _get_spam_scams_patterns(self) -> Dict[str, List[str]]:
        return {
              "financial_scams": [
                r'(?i)\bmake\s+\$?\d+\s+(from\s+home|working\s+from\s+home|per\s+day)',
                r'(?i)\bget\s+rich\s+quick',
                r'(?i)\b(investment|trading)\s+opportunity\s+of\s+a\s+lifetime',
                r'(?i)\bdouble\s+your\s+money\s+in\s+\d+\s+(days|weeks)',
                r'(?i)\bnigerian\s+prince\s+(scam|email|inheritance)',
                r'(?i)\badvance\s+fee\s+(fraud|scam)',
            ],
            "crypto_scams": [
                r'(?i)\bfree\s+(bitcoin|cryptocurrency|crypto)',
                r'(?i)\bcrypto\s+(giveaway|airdrop)\s+scam',
                r'(?i)\binvest\s+in\s+.*(shitcoin|meme\s+coin)\s+now',
                r'(?i)\brug\s+pull\s+(crypto|token|coin)',
                r'(?i)\bpump\s+and\s+dump\s+(scheme|crypto|token)',
            ],
            "phishing": [
                r'(?i)\bclick\s+here\s+to\s+(claim|get|receive)\s+your',
                r'(?i)\byour\s+account\s+(will\s+be\s+)?(suspended|closed|deleted)',
                r'(?i)\bverify\s+your\s+(account|identity|information)\s+now',
                r'(?i)\bupdate\s+your\s+(payment|billing)\s+information',
                r'(?i)\bsecurity\s+alert\s+(click|verify|update)',
            ],
            "mlm_schemes": [
                r'(?i)\bbecome\s+(your\s+own\s+)?boss',
                r'(?i)\b(join\s+my\s+team|work\s+from\s+home\s+opportunity)',
                r'(?i)\bunlimited\s+(income|earning)\s+potential',
                r'(?i)\bpyramid\s+scheme\s+(disguised\s+as|mlm)',
                r'(?i)\bmulti\s+level\s+marketing\s+(scam|scheme)',
            ],
            "general_scamming": [
                r'(?i)\bhow\s+to\s+(scam|con|trick|deceive|swindle)\s+(people|someone|others)',
                r'(?i)\b(scam|con|trick|deceive|swindle)\s+(people|someone|others)\s+(out\s+of|for)',
                r'(?i)\b(easy|quick)\s+(scam|con)\s+(method|technique|way)',
                r'(?i)\bscamming\s+(techniques|methods|strategies)',
                r'(?i)\bhow\s+to\s+(rip\s+off|take\s+advantage\s+of)\s+(people|customers|clients)',
                r'(?i)\b(ponzi|pyramid)\s+scheme\s+(guide|instructions)',
                r'(?i)\bfraud\s+(people|customers|clients|system)',
                r'(?i)\binsurance\s+fraud\s+(claim|method|scheme)',
                r'(?i)\btax\s+(evasion|fraud)\s+(method|scheme)',
            ]
        }
    
    def _get_privacy_patterns(self) -> Dict[str, List[str]]:
        return {
            "pii_request": [
                r'(?i)\b(give|tell|share)\s+me\s+your\s+(name|address|phone|email)',
                r'(?i)\bwhat\s+is\s+your\s+(real\s+name|address|location)',
                r'(?i)\bshare\s+your\s+(personal|private)\s+information',
            ],
            "doxxing_attempt": [
                r'(?i)\bfind\s+(someone\'s|a\s+person\'s)\s+(address|phone|information)',
                r'(?i)\bhow\s+to\s+(stalk|track|find)\s+someone\s+online',
                r'(?i)\bget\s+someone\'s\s+(personal|private)\s+(info|information|data)',
            ],
            "data_harvesting": [
                r'(?i)\bcollect\s+(user|personal)\s+data\s+without\s+consent',
                r'(?i)\bscrape\s+(social\s+media|website)\s+for\s+(data|information)',
                r'(?i)\bharvest\s+(email|contact)\s+(addresses|information)',
            ]
        }
    
    def _get_malicious_uri_patterns(self) -> Dict[str, List[str]]:
        return {
            "suspicious_domains": [
                r'(?i)\b(bit\.ly|tinyurl|goo\.gl|t\.co)\/\w+',
                r'(?i)\b\w+\.(tk|ml|ga|cf)\b',
                r'(?i)\b(phishing|malware|virus)\.(com|net|org)',
            ],
            "malicious_content": [
                r'(?i)\bdownload\s+.*(virus|malware|trojan|keylogger)',
                r'(?i)\b(click|visit)\s+this\s+link\s+to\s+(hack|steal|scam)',
                r'(?i)\bmalicious\s+(url|link|website)',
            ],
            "ip_addresses": [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                r'(?i)\bconnect\s+to\s+ip\s+address',
            ]
        }
    
    def _get_edge_cases_patterns(self) -> Dict[str, List[str]]:
        return {
            "context_dependent_profanity": [

                r'(?i)\b(fuck|fucking|shit|damn|hell|ass|bitch)\b',
                r'(?i)\b(stupid|idiot|moron|retard|dumb)\b',
                r'(?i)\b(gay|fag|homo)\b',
            ],
            "legitimate_educational": [

                r'(?i)\b(sex\s+education|sexual\s+health|anatomy)',
                r'(?i)\b(violence\s+in\s+history|war\s+crimes|genocide\s+studies)',
                r'(?i)\b(drug\s+education|substance\s+abuse\s+prevention)',
            ],
            "medical_terms": [

                r'(?i)\b(breast|penis|vagina|anus|sexual\s+dysfunction)',
                r'(?i)\b(suicide\s+prevention|mental\s+health|depression)',
                r'(?i)\b(overdose|addiction|withdrawal)',
            ],
            "creative_content": [

                r'(?i)\bwrite\s+a\s+(story|novel|script)\s+(about|involving)',
                r'(?i)\b(fictional|fantasy|sci-fi)\s+(violence|war|conflict)',
                r'(?i)\b(character|protagonist)\s+(kills|murders|fights)',
            ],
            "news_and_reporting": [

                r'(?i)\breporting\s+on\s+(violence|crime|terrorism)',
                r'(?i)\bnews\s+(article|report)\s+about',
                r'(?i)\bjournalism\s+(investigation|coverage)',
            ],
            "legitimate_religious_discussion": [

                r'(?i)\b(religious\s+studies|theology|comparative\s+religion|interfaith\s+dialogue)',
                r'(?i)\b(academic\s+study|scholarly\s+analysis|research)\s+.*(religion|islam|christianity|judaism|hinduism|buddhism)',
                r'(?i)\b(history\s+of|origins\s+of|development\s+of)\s+(religion|christianity|islam|judaism)',
                r'(?i)\b(biblical|quranic|talmudic|vedic)\s+(studies|scholarship|analysis|interpretation)',
                r'(?i)\b(religious\s+freedom|freedom\s+of\s+religion|religious\s+rights|religious\s+tolerance)',
                r'(?i)\b(understanding|learning\s+about|exploring)\s+(different\s+)?(religions|faiths|beliefs)',
                r'(?i)\b(peace\s+between|dialogue\s+between|cooperation\s+between)\s+(religions|faiths)',
                r'(?i)\b(teaching\s+about|education\s+about)\s+(world\s+religions|religious\s+diversity)',
                r'(?i)\b(religious\s+art|religious\s+music|religious\s+literature|religious\s+architecture)',
                r'(?i)\b(pilgrimage|prayer|meditation|worship|ritual)\s+(practices|traditions|customs)',
                r'(?i)\b(seminary|divinity\s+school|religious\s+education|sunday\s+school|madrasa)',
                r'(?i)\b(pastor|priest|rabbi|imam|monk|guru)\s+(training|education|role|responsibilities)',
            ]
        }

# =============================================================================
# Advanced Detection Engine
# =============================================================================

@dataclass
class RuleBasedConfig:
    """Enhanced configuration with comprehensive settings"""
    
    # Input limits
    max_input_length: int = 10000
    max_word_count: int = 2000
    max_line_count: int = 100
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    burst_limit: int = 10
    burst_window: int = 60
    
    # Output limits
    max_output_length: int = 50000
    max_output_sentences: int = 100
    
    # Detection thresholds
    profanity_threshold: float = 0.6
    hate_speech_threshold: float = 0.3
    violence_threshold: float = 0.4
    harassment_threshold: float = 0.5
    dangerous_content_threshold: float = 0.7
    
    # Context sensitivity
    enable_context_analysis: bool = True
    educational_context_weight: float = 0.3
    medical_context_weight: float = 0.4
    creative_context_weight: float = 0.5
    
    # Edge case handling
    enable_false_positive_reduction: bool = True
    enable_severity_adjustment: bool = True

class GuardrailViolation(BaseModel):
    violation_type: ViolationType
    severity: Severity
    message: str
    pattern_matched: Optional[str] = None
    confidence_score: float = 1.0
    context_detected: Optional[ContextType] = None
    suggested_action: str
    location: Optional[str] = None
    
    class Config:
        use_enum_values = True

class LLMRequest(BaseModel):
    prompt: str = Field(..., max_length=50000)
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=150, ge=1, le=4000)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context_hint: Optional[str] = None

class LLMResponse(BaseModel):
    response: str
    filtered: bool = False
    confidence_score: float = 1.0
    violations: List[GuardrailViolation] = []
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        use_enum_values = True

# =============================================================================
# Context-Aware Detection Engine
# =============================================================================

class ContextAnalyzer:
    """Analyzes text context to reduce false positives"""
    
    def __init__(self, patterns: ComprehensivePatternLibrary):
        self.patterns = patterns
    
    def detect_context(self, text: str) -> Optional[ContextType]:
        """Detect the likely context of the text"""
        text_lower = text.lower()
        
        # Educational context
        educational_score = sum(1 for indicator in self.patterns.educational_indicators 
                              if indicator in text_lower)
        
        # Medical context
        medical_score = sum(1 for indicator in self.patterns.medical_indicators 
                           if indicator in text_lower)
        
        # Creative context
        creative_score = sum(1 for indicator in self.patterns.creative_indicators 
                            if indicator in text_lower)
        
        # News context
        news_score = sum(1 for indicator in self.patterns.news_indicators 
                        if indicator in text_lower)
        
        # Religious academic context (counts as educational)
        religious_academic_score = sum(1 for indicator in self.patterns.religious_academic_indicators 
                                     if indicator in text_lower)
        
        # Add religious academic score to educational score
        educational_score += religious_academic_score
        
        scores = {
            ContextType.EDUCATIONAL: educational_score,
            ContextType.MEDICAL: medical_score,
            ContextType.CREATIVE: creative_score,
            ContextType.NEWS: news_score,
        }
        
        max_score = max(scores.values())
        if max_score >= 2:
            return max(scores, key=scores.get)
        
        return ContextType.CASUAL
    
    def adjust_severity_for_context(self, severity: Severity, context: ContextType, 
                                   violation_type: ViolationType) -> Severity:
        """Adjust violation severity based on context"""
        

        if context == ContextType.EDUCATIONAL:
            if violation_type in [ViolationType.VIOLENCE, ViolationType.DANGEROUS]:
                if severity == Severity.HIGH:
                    return Severity.MEDIUM
                elif severity == Severity.MEDIUM:
                    return Severity.LOW
            

            if violation_type == ViolationType.HATE_SPEECH:

                if severity == Severity.CRITICAL:
                    return Severity.HIGH
                elif severity == Severity.HIGH:
                    return Severity.MEDIUM
        

        if context == ContextType.MEDICAL:
            if violation_type == ViolationType.SEXUALLY_EXPLICIT:
                if severity in [Severity.MEDIUM, Severity.LOW]:
                    return Severity.LOW
        

        if context == ContextType.CREATIVE:
            if violation_type in [ViolationType.VIOLENCE, ViolationType.HARASSMENT]:
                if severity == Severity.HIGH:
                    return Severity.MEDIUM
                elif severity == Severity.MEDIUM:
                    return Severity.LOW
        

        if context == ContextType.NEWS:
            if violation_type in [ViolationType.VIOLENCE, ViolationType.HATE_SPEECH]:
                if severity == Severity.HIGH:
                    return Severity.MEDIUM
                elif severity == Severity.MEDIUM:
                    return Severity.LOW
        
        return severity

class ComprehensiveDetector:
    """Main detection engine with comprehensive pattern matching"""
    
    def __init__(self, config: RuleBasedConfig):
        self.config = config
        self.patterns = ComprehensivePatternLibrary()
        self.context_analyzer = ContextAnalyzer(self.patterns)
        

        self._compile_patterns()
        

        self.compiled_pii_patterns = {
            name: re.compile(pattern, re.IGNORECASE) 
            for name, pattern in self.patterns.pii_patterns.items()
        }
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficient matching"""
        self.compiled_patterns = {}
        
        for category in ['prompt_injection', 'jailbreak', 'hate_speech', 'harassment', 
                        'violence', 'sexually_explicit', 'dangerous', 'misinformation',
                        'spam_scams', 'privacy', 'malicious_uri', 'edge_cases']:
            
            category_patterns = getattr(self.patterns, category)
            self.compiled_patterns[category] = {}
            
            for subcategory, pattern_list in category_patterns.items():
                self.compiled_patterns[category][subcategory] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in pattern_list
                ]
    
    def detect_violations(self, text: str, context_hint: str = None) -> List[GuardrailViolation]:
        """Comprehensive violation detection with context awareness"""
        violations = []
        

        detected_context = self.context_analyzer.detect_context(text)
        if context_hint:

            try:
                detected_context = ContextType(context_hint.lower())
            except ValueError:
                pass
        

        violations.extend(self._detect_category(text, 'prompt_injection', ViolationType.PROMPT_INJECTION, Severity.HIGH))
        violations.extend(self._detect_category(text, 'jailbreak', ViolationType.JAILBREAK, Severity.HIGH))
        violations.extend(self._detect_category(text, 'hate_speech', ViolationType.HATE_SPEECH, Severity.CRITICAL))
        violations.extend(self._detect_category(text, 'harassment', ViolationType.HARASSMENT, Severity.HIGH))
        violations.extend(self._detect_category(text, 'violence', ViolationType.VIOLENCE, Severity.HIGH))
        violations.extend(self._detect_category(text, 'sexually_explicit', ViolationType.SEXUALLY_EXPLICIT, Severity.HIGH))
        violations.extend(self._detect_category(text, 'dangerous', ViolationType.DANGEROUS, Severity.CRITICAL))
        violations.extend(self._detect_category(text, 'misinformation', ViolationType.MISINFORMATION, Severity.MEDIUM))
        violations.extend(self._detect_category(text, 'spam_scams', ViolationType.SPAM_SCAMS, Severity.MEDIUM))
        violations.extend(self._detect_category(text, 'privacy', ViolationType.PRIVACY, Severity.HIGH))
        violations.extend(self._detect_category(text, 'malicious_uri', ViolationType.MALICIOUS_URI, Severity.HIGH))
        

        violations.extend(self._detect_pii(text))
        

        violations.extend(self._detect_edge_cases(text, detected_context))
        

        if self.config.enable_severity_adjustment:
            for violation in violations:
                violation.context_detected = detected_context
                original_severity = Severity(violation.severity)
                adjusted_severity = self.context_analyzer.adjust_severity_for_context(
                    original_severity, detected_context, violation.violation_type
                )
                violation.severity = adjusted_severity
        
        return violations
    
    def _detect_pii(self, text: str) -> List[GuardrailViolation]:
        """Detect personally identifiable information"""
        violations = []
        
        for pii_type, pattern in self.compiled_pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                first_match = str(matches[0]) if matches else ""
                if matches and isinstance(matches[0], tuple):
                    first_match = ' '.join(str(x) for x in matches[0] if x)
                
                violations.append(GuardrailViolation(
                    violation_type=ViolationType.PII_DETECTED,
                    severity=Severity.HIGH,
                    message=f"PII detected: {pii_type.upper()} ({len(matches)} instances)",
                    pattern_matched=pattern.pattern,
                    confidence_score=1.0,
                    suggested_action="Redact PII",
                    location=f"First match: {first_match[:20]}..." if first_match else None
                ))
        
        return violations
    
    def _detect_category(self, text: str, category: str, violation_type: ViolationType, 
                        default_severity: Severity) -> List[GuardrailViolation]:
        """Detect violations for a specific category"""
        violations = []
        
        category_patterns = self.compiled_patterns.get(category, {})
        
        for subcategory, patterns in category_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:

                    first_match = str(matches[0]) if matches else ""
                    if matches and isinstance(matches[0], tuple):
                        first_match = ' '.join(str(x) for x in matches[0] if x)
                    

                    actual_severity = default_severity
                    if category == "dangerous":
                            if subcategory == "explosives":
                                actual_severity = Severity.CRITICAL
                            elif subcategory == "drug_trafficking":
                                actual_severity = Severity.CRITICAL
                            elif subcategory == "aviation_terrorism":
                                actual_severity = Severity.CRITICAL
                            elif subcategory == "mass_shooting":
                                actual_severity = Severity.CRITICAL
                            elif subcategory == "assassination":
                                actual_severity = Severity.CRITICAL
                            elif subcategory == "indiscriminate_weapons":
                                actual_severity = Severity.CRITICAL
                        
                    
                    violations.append(GuardrailViolation(
                        violation_type=violation_type,
                        severity=actual_severity,
                        message=f"{violation_type.value.replace('_', ' ').title()} detected in {subcategory}",
                        pattern_matched=pattern.pattern,
                        confidence_score=self._calculate_confidence(matches, text),
                        suggested_action=self._get_suggested_action(violation_type, actual_severity),
                        location=f"Match: {first_match[:50]}..." if first_match else None
                    ))
        
        return violations
    
    def _detect_edge_cases(self, text: str, context: ContextType) -> List[GuardrailViolation]:
        """Special handling for edge cases with context awareness"""
        violations = []
        

        profanity_patterns = self.compiled_patterns['edge_cases']['context_dependent_profanity']
        
        for pattern in profanity_patterns:
            matches = pattern.findall(text)
            if matches:

                word = matches[0].lower() if matches else ""
                severity = self._get_profanity_severity(word, context)
                
                if severity != Severity.LOW or context == ContextType.CASUAL:
                    violations.append(GuardrailViolation(
                        violation_type=ViolationType.EDGE_CASES,
                        severity=severity,
                        message=f"Context-dependent language detected: {word}",
                        pattern_matched=pattern.pattern,
                        confidence_score=0.7,
                        context_detected=context,
                        suggested_action=self._get_suggested_action(ViolationType.EDGE_CASES, severity),
                        location=f"Word: {word}"
                    ))
        
        return violations
    
    def _get_profanity_severity(self, word: str, context: ContextType) -> Severity:
        """Determine profanity severity based on word and context"""
        

        if word in ['fuck', 'fucking', 'shit']:
            if context in [ContextType.EDUCATIONAL, ContextType.MEDICAL, ContextType.NEWS]:
                return Severity.LOW
            elif context == ContextType.CREATIVE:
                return Severity.MEDIUM
            else:
                return Severity.HIGH
        

        elif word in ['damn', 'hell', 'ass', 'bitch']:
            if context in [ContextType.EDUCATIONAL, ContextType.MEDICAL, ContextType.NEWS, ContextType.CREATIVE]:
                return Severity.LOW
            else:
                return Severity.MEDIUM
        

        elif word in ['stupid', 'idiot', 'moron', 'dumb']:
            if context in [ContextType.EDUCATIONAL, ContextType.MEDICAL]:
                return Severity.LOW
            else:
                return Severity.MEDIUM
        

        elif word in ['gay', 'fag', 'homo']:
            if context in [ContextType.EDUCATIONAL, ContextType.MEDICAL, ContextType.NEWS]:
                return Severity.LOW
            else:
                return Severity.HIGH
        
        return Severity.MEDIUM
    
    def _calculate_confidence(self, matches: List, text: str) -> float:
        """Calculate confidence score for violation"""
        if not matches:
            return 0.0
        

        base_score = min(1.0, len(matches) / 3)
        length_factor = min(1.0, len(text) / 1000)
        
        return max(0.1, base_score * (0.7 + 0.3 * length_factor))
    
    def _get_suggested_action(self, violation_type: ViolationType, severity: Severity) -> str:
        """Get suggested action based on violation type and severity"""
        if severity == Severity.CRITICAL:
            return "Block immediately"
        elif severity == Severity.HIGH:
            return "Block or heavily filter"
        elif severity == Severity.MEDIUM:
            return "Review and potentially filter"
        else:
            return "Monitor or flag"

# =============================================================================
# Rate Limiting System
# =============================================================================

class AdvancedRateLimiter:
    """Advanced rate limiting with burst detection"""
    
    def __init__(self, config: RuleBasedConfig):
        self.config = config
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.burst_requests: Dict[str, deque] = defaultdict(deque)
        self.violation_counts: Dict[str, int] = defaultdict(int)
    
    def is_rate_limited(self, identifier: str) -> Tuple[bool, str]:
        """Check if request should be rate limited"""
        now = time.time()
        

        burst_window_start = now - self.config.burst_window
        burst_requests = self.burst_requests[identifier]
        
        while burst_requests and burst_requests[0] < burst_window_start:
            burst_requests.popleft()
        
        if len(burst_requests) >= self.config.burst_limit:
            return True, f"Burst limit exceeded: {len(burst_requests)}/{self.config.burst_limit}"
        

        window_start = now - self.config.rate_limit_window
        user_requests = self.requests[identifier]
        
        while user_requests and user_requests[0] < window_start:
            user_requests.popleft()
        
        if len(user_requests) >= self.config.rate_limit_requests:
            return True, f"Rate limit exceeded: {len(user_requests)}/{self.config.rate_limit_requests}"
        

        burst_requests.append(now)
        user_requests.append(now)
        
        return False, "OK"

# =============================================================================
# Main Guardrails Server
# =============================================================================

class ComprehensiveGuardrailsServer:
    """Comprehensive guardrails server with edge case handling"""
    
    def __init__(self, config: RuleBasedConfig = None):
        self.config = config or RuleBasedConfig()
        self.rate_limiter = AdvancedRateLimiter(self.config)
        self.detector = ComprehensiveDetector(self.config)
        

        self.stats = {
            'total_requests': 0,
            'blocked_inputs': 0,
            'filtered_outputs': 0,
            'rate_limited': 0,
            'violations_by_type': defaultdict(int),
            'context_adjustments': 0,
            'false_positives_prevented': 0,
            'religious_abuse_blocked': 0,
            'hate_speech_blocked': 0,
            'harassment_blocked': 0,
        }
        

        self.audit_log: deque = deque(maxlen=10000)
        

        self.app = FastAPI(
            title="Comprehensive ModelShield Server",
            description="Complete guardrails with edge case handling and context awareness",
            version="3.0.0"
        )
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _violations_to_dict(self, violations: List[GuardrailViolation]) -> List[Dict]:
        """Safely convert violations to dictionary format for JSON serialization"""
        result = []
        for v in violations:
            try:

                violation_type = v.violation_type.value if hasattr(v.violation_type, 'value') else str(v.violation_type)
                

                severity = v.severity.value if hasattr(v.severity, 'value') else str(v.severity)
                
                result.append({
                    "violation_type": violation_type,
                    "severity": severity,
                    "message": v.message,
                    "pattern_matched": v.pattern_matched,
                    "confidence_score": v.confidence_score,
                    "suggested_action": v.suggested_action,
                    "location": v.location
                })
            except Exception as e:

                logger.warning(f"Error processing violation: {str(e)}")
                result.append({
                    "violation_type": "error",
                    "severity": "medium", 
                    "message": f"Error processing violation: {str(e)}",
                    "confidence_score": 0.0,
                    "suggested_action": "Review manually"
                })
        
        return result
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0",
                "features": [
                    "comprehensive_patterns", "context_awareness", "edge_case_handling",
                    "false_positive_reduction", "severity_adjustment"
                ]
            }
        
        @self.app.get("/patterns")
        async def get_patterns():
            """Get information about all detection patterns"""
            pattern_counts = {}
            
            for category in ['prompt_injection', 'jailbreak', 'hate_speech', 'harassment',
                           'violence', 'sexually_explicit', 'dangerous', 'misinformation',
                           'spam_scams', 'privacy', 'malicious_uri', 'edge_cases']:
                category_patterns = getattr(self.detector.patterns, category)
                pattern_counts[category] = {
                    "subcategories": len(category_patterns),
                    "total_patterns": sum(len(patterns) for patterns in category_patterns.values()),
                    "patterns": category_patterns
                }
            
            return {
                "total_categories": len(pattern_counts),
                "total_patterns": sum(pc["total_patterns"] for pc in pattern_counts.values()),
                "categories": pattern_counts,
                "context_types": [ct.value for ct in ContextType],
                "violation_types": [vt.value for vt in ViolationType]
            }
        
        @self.app.post("/debug-detection")
        async def debug_detection(request: Request, llm_request: LLMRequest):
            """Debug endpoint to see what patterns match without blocking"""
            user_id = llm_request.user_id or request.client.host
            

            violations = self.detector.detect_violations(
                llm_request.prompt, 
                llm_request.context_hint
            )
            

            debug_info = {}
            for category in ['dangerous', 'edge_cases', 'hate_speech', 'harassment']:
                category_violations = self.detector._detect_category(
                    llm_request.prompt, 
                    category, 
                    getattr(ViolationType, category.upper()), 
                    Severity.HIGH
                )
                debug_info[category] = {
                    "violations_count": len(category_violations),
                    "violations": self._violations_to_dict(category_violations)
                }
            
            return {
                "input": llm_request.prompt,
                "total_violations": len(violations),
                "violations": self._violations_to_dict(violations),
                "debug_by_category": debug_info,
                "would_be_blocked": len([v for v in violations if v.severity in [Severity.HIGH, Severity.CRITICAL]]) > 0
            }
        
        @self.app.post("/validate-input")
        async def validate_input_endpoint(request: Request, llm_request: LLMRequest):
            start_time = time.time()
            client_ip = request.client.host
            user_id = llm_request.user_id or client_ip
            
            self.stats['total_requests'] += 1
            

            is_limited, limit_reason = self.rate_limiter.is_rate_limited(user_id)
            if is_limited:
                self.stats['rate_limited'] += 1
                self._log_event("rate_limit_exceeded", user_id, {"reason": limit_reason})
                raise HTTPException(status_code=429, detail=f"Rate limited: {limit_reason}")
            

            violations = self.detector.detect_violations(
                llm_request.prompt, 
                llm_request.context_hint
            )
            
            processing_time = time.time() - start_time

            for violation in violations:

                violation_type = violation.violation_type.value if hasattr(violation.violation_type, 'value') else str(violation.violation_type)
                self.stats['violations_by_type'][violation_type] += 1
                if violation.context_detected and violation.context_detected != ContextType.CASUAL:
                    self.stats['context_adjustments'] += 1
                

                if violation.violation_type == ViolationType.HATE_SPEECH:
                    self.stats['hate_speech_blocked'] += 1

                    if any(keyword in violation.message.lower() for keyword in ['religious', 'muslim', 'jew', 'christian', 'hindu', 'buddhist', 'islam', 'christianity', 'judaism']):
                        self.stats['religious_abuse_blocked'] += 1
                elif violation.violation_type == ViolationType.HARASSMENT:
                    self.stats['harassment_blocked'] += 1

                    if any(keyword in violation.message.lower() for keyword in ['religious', 'muslim', 'jew', 'christian', 'hindu', 'buddhist', 'islam', 'christianity', 'judaism']):
                        self.stats['religious_abuse_blocked'] += 1
                elif violation.violation_type == ViolationType.DANGEROUS:

                    if any(keyword in violation.message.lower() for keyword in ['drug', 'cocaine', 'heroin', 'meth', 'trafficking', 'dealer']):
                        self.stats['drug_trafficking_blocked'] += 1
                elif violation.violation_type == ViolationType.EDGE_CASES:

                    if any(keyword in violation.message.lower() for keyword in ['profanity', 'context-dependent', 'language']):
                        self.stats['profanity_detected'] += 1
            

            critical_violations = [v for v in violations if v.severity in [Severity.HIGH, Severity.CRITICAL]]
            is_blocked = len(critical_violations) > 0
            

            hate_speech_violations = [v for v in violations if v.violation_type == ViolationType.HATE_SPEECH]
            if hate_speech_violations:
                is_blocked = True
            
            if is_blocked:
                self.stats['blocked_inputs'] += 1
            
            self._log_event("input_validation", user_id, {
                "blocked": is_blocked,
                "violations": len(violations),
                "processing_time_ms": processing_time * 1000
            })
            
            if is_blocked:
                raise HTTPException(status_code=400, detail={
                    "message": "Input validation failed",
                    "violations": self._violations_to_dict(violations),
                    "processing_time_ms": processing_time * 1000
                })
            
            return {
                "valid": True,
                "violations": self._violations_to_dict(violations),
                "warnings": [v.message for v in violations if v.severity in [Severity.LOW, Severity.MEDIUM]],
                "processing_time_ms": processing_time * 1000,
                "detection_method": "comprehensive_rule_based",
                "context_analysis_enabled": self.config.enable_context_analysis
            }
        
        @self.app.get("/stats")
        async def get_stats():
            total = max(self.stats['total_requests'], 1)
            return {
                **dict(self.stats),
                "success_rate": (total - self.stats['blocked_inputs']) / total,
                "active_users": len(self.rate_limiter.requests),
                "detection_categories": len(ViolationType),
                "context_types_supported": len(ContextType),
                "comprehensive": True
            }
    
    def _log_event(self, event_type: str, user_id: str, details: Dict):
        """Log audit event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details
        }
        self.audit_log.append(event)
        logger.info(f"Event: {event_type} for user {user_id}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the guardrails server"""
        logger.info(f"Starting  ModelShield Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

# =============================================================================
# Usage and Configuration
# =============================================================================

def create_production_config() -> RuleBasedConfig:
    """Create production-ready configuration"""
    config = RuleBasedConfig()
    

    config.profanity_threshold = 0.4
    config.hate_speech_threshold = 0.2
    config.violence_threshold = 0.3
    config.harassment_threshold = 0.3
    config.dangerous_content_threshold = 0.5
    

    config.enable_context_analysis = True
    config.enable_false_positive_reduction = True
    config.enable_severity_adjustment = True
    

    config.rate_limit_requests = 500
    config.burst_limit = 3000
    
    return config

if __name__ == "__main__":
    config = create_production_config()
    server = ComprehensiveGuardrailsServer(config)
    
    print("🛡️ ModelShield SERVER ")
    print("=" * 80)
    print("🎯 COMPREHENSIVE COVERAGE:")
    print("  ✅ Prompt Injection & Jailbreaking")
    print("  ✅ Hate Speech & Harassment") 
    print("  ✅ Violence & Dangerous Content")
    print("  ✅ Sexually Explicit Content")
    print("  ✅ Misinformation & Spam/Scams")
    print("  ✅ Privacy & Malicious URIs")
    print("  ✅ Edge Cases & Context Awareness")
    print()
    print("🧠 SMART FEATURES:")
    print("  ✅ Context-Aware Detection (Educational/Medical/Creative/News)")
    print("  ✅ Profanity Severity Adjustment (fuck/stupid/idiot handled properly)")
    print("  ✅ False Positive Reduction")
    print("  ✅ Severity Adjustment Based on Context")
    print("  ✅ Comprehensive Edge Case Handling")
    print()
    print("📊 COVERAGE STATS:")
    patterns = ComprehensivePatternLibrary()
    total_patterns = 0
    for category in ['prompt_injection', 'jailbreak', 'hate_speech', 'harassment',
                    'violence', 'sexually_explicit', 'dangerous', 'misinformation',
                    'spam_scams', 'privacy', 'malicious_uri', 'edge_cases']:
        category_patterns = getattr(patterns, category)
        category_total = sum(len(pattern_list) for pattern_list in category_patterns.values())
        total_patterns += category_total
        print(f"  • {category.replace('_', ' ').title()}: {category_total} patterns")
    

    pii_total = len(patterns.pii_patterns)
    total_patterns += pii_total
    print(f"  • PII Detection: {pii_total} patterns")
    
    print(f"\n  🎯 TOTAL PATTERNS: {total_patterns}")
    print("=" * 80)
    print(f"🚀 Server starting on http://localhost:8000")
    print("📖 API docs: http://localhost:8000/docs")
    print("🔍 Patterns: http://localhost:8000/patterns")
    print()
    
    server.run()