import hashlib
import diskcache as dc

_CACHES = {}
def get_cache(provider):
    if provider not in _CACHES:
        _CACHES[provider] = dc.Cache(f'cache/{provider}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')
    return _CACHES[provider]


def cache_key(model, request):
    raw = f"{model}_{request['source_language']}_{request['target_language']}_{request['prompt']}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()
