import requests

class EuTrialClient:
    BASE = 'https://clinicaltrials.gov/api/v2/studies'

    def search(self, q, top_k=5):
        params = {'query.term': q, 'pageSize': top_k, 'filter.geo': 'distance(50.1109,8.6821,5000km)'}
        try:
            resp = requests.get(self.BASE, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return [{'error': str(exc)}]
        out = []
        for study in data.get('studies', [])[:top_k]:
            proto = study.get('protocolSection', {})
            ident = proto.get('identificationModule', {})
            desc = proto.get('descriptionModule', {})
            out.append({'id': ident.get('nctId'), 'title': ident.get('briefTitle'), 'text': desc.get('briefSummary', ''), 'source_type': 'eu_trial'})
        return out
