import logging
import os
import re
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)

PII_PATTERNS = [re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)]
MEDICAL_ADVICE_PATTERNS = [re.compile(r"\b(should i|diagnose|prescribe|dosage)\b", re.I)]
OUT_OF_SCOPE_PATTERNS = [re.compile(r"\b(stock price|weather|recipe)\b", re.I)]
EU_HINT_PATTERNS = [re.compile(r"\b(GDPR|MDR|IVDR|EMA|EU|European)\b", re.I)]

class ScopeCategory(str, Enum):
    in_scope = "in_scope"
    out_of_scope = "out_of_scope"
    pii_phi_request = "pii_phi_request"
    medical_advice_request = "medical_advice_request"

class ScopeDecision(BaseModel):
    category: ScopeCategory
    reason: str
    jurisdiction_hint: str = Field(default="unknown")

def detect_pii(text: str) -> bool:
    return any(p.search(text or '') for p in PII_PATTERNS)

def _jurisdiction_hint(query: str) -> str:
    return 'EU' if any(p.search(query or '') for p in EU_HINT_PATTERNS) else 'unknown'

def _log_gdpr_pii(query: str, reason: str) -> None:
    try:
        from monitoring.database import log_compliance_violation
        log_compliance_violation(violation_type='GDPR_PII', query_excerpt=(query or '')[:500], jurisdiction_hint=_jurisdiction_hint(query), details=reason)
    except Exception as exc:
        logger.warning('Could not log GDPR_PII violation: %s', exc)

def check_scope_guard(query: str, model: Optional[str] = None) -> ScopeDecision:
    text = (query or '').strip()
    jurisdiction_hint = _jurisdiction_hint(text)
    if detect_pii(text):
        reason = 'Request appears to include or seek personal identifiers.'
        _log_gdpr_pii(text, reason)
        return ScopeDecision(category=ScopeCategory.pii_phi_request, reason=reason, jurisdiction_hint=jurisdiction_hint)
    if any(p.search(text) for p in MEDICAL_ADVICE_PATTERNS):
        return ScopeDecision(category=ScopeCategory.medical_advice_request, reason='Individual medical advice is out of scope.', jurisdiction_hint=jurisdiction_hint)
    if any(p.search(text) for p in OUT_OF_SCOPE_PATTERNS):
        return ScopeDecision(category=ScopeCategory.out_of_scope, reason='Outside EU healthcare law scope.', jurisdiction_hint=jurisdiction_hint)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return ScopeDecision(category=ScopeCategory.in_scope, reason='Heuristic in-scope.', jurisdiction_hint=jurisdiction_hint)
    client = OpenAI(api_key=api_key)
    model = model or os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-4o-mini')
    system = 'EU medico-legal scope guard. Categories: in_scope, out_of_scope, pii_phi_request, medical_advice_request. jurisdiction_hint EU or unknown.'
    try:
        completion = client.beta.chat.completions.parse(model=model, messages=[{'role':'system','content':system},{'role':'user','content':text}], response_format=ScopeDecision, temperature=0)
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            if parsed.category == ScopeCategory.pii_phi_request:
                _log_gdpr_pii(text, parsed.reason)
            return parsed
    except Exception as exc:
        logger.warning('LLM scope guard failed: %s', exc)
    return ScopeDecision(category=ScopeCategory.in_scope, reason='Default allow for EU medico-legal queries.', jurisdiction_hint=jurisdiction_hint)

scope_guard = check_scope_guard
