from pydantic import BaseModel, Field
from typing import List
import os
import json
from llm.provider import chat_completion, extract_json_object, get_provider, resolve_model
from openai import OpenAI

CRITERIA = ['factual_accuracy','citation_quality','eu_jurisdiction_alignment','regulatory_compliance','privacy_gdpr','clarity','completeness','harm_avoidance','source_recency','bias_fairness','actionable_guidance']

class CriterionScore(BaseModel):
    name: str
    score: int = Field(ge=1, le=5)
    rationale: str

class JudgeResult(BaseModel):
    criteria: List[CriterionScore]
    overall: float = 0.0

def llm_judge(question, answer, context=None, model=None):
    provider = get_provider()
    prompt = 'Question: ' + str(question) + '\nAnswer: ' + str(answer) + '\nContext: ' + str(context or '')
    if provider in ('huggingface', 'stub'):
        system = (
            'Score EU medico-legal RAG answers. Reply JSON only: '
            '{"criteria":[{"name":"factual_accuracy","score":3,"rationale":"..."}],"overall":3.0}'
        )
        raw = chat_completion(system=system, user=prompt, model=resolve_model(model), temperature=0)
        try:
            data = extract_json_object(raw)
            return JudgeResult.model_validate(data)
        except Exception:
            return JudgeResult(criteria=[], overall=0.0)
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model = model or os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-4o-mini')
    completion = client.beta.chat.completions.parse(model=model, messages=[{'role':'system','content':'Score EU medico-legal RAG answers.'},{'role':'user','content':prompt}], response_format=JudgeResult, temperature=0)
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return JudgeResult(criteria=[], overall=0.0)
    if not parsed.overall:
        total = sum(c.score for c in parsed.criteria)
        parsed.overall = total / max(len(parsed.criteria), 1)
    return parsed
