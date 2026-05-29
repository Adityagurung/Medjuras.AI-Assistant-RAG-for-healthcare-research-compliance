from dotenv import load_dotenv
load_dotenv()
from search.hybrid_search import eu_hybrid_search
from tools.pubmed import PubMedTool
from tools.wikipedia import WikipediaTool
import importlib
_ct = importlib.import_module('tools.' + 'clinicaltrials')

PUBMED = PubMedTool()
WIKI = WikipediaTool()
CT = _ct.EuTrialClient()

def tool_eu_hybrid_search(q, top_k=5, local=False):
    docs = eu_hybrid_search(q, top_k=top_k, local=local)
    return [{ 'id': d.id, 'title': d.title, 'text': d.text, 'rrf_score': float(getattr(d,'rrf_score',0.0)), 'source_type': getattr(d,'source_type',None)} for d in docs]

FUNCTION_MAP = {
    'eu_hybrid_search': tool_eu_hybrid_search,
    'pubmed_search': PUBMED.pubmed_semantic_search,
    'wikipedia_search': WIKI.wiki_semantic_search,
    'clinicaltrials_search': CT.search,
}

TOOLS_JSON = [{'type':'function','function':{'name':'eu_hybrid_search','parameters':{'type':'object','properties':{'q':{'type':'string'}},'required':['q']}}}]
