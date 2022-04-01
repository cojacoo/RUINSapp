from typing import Callable, Dict

DEFAULT = {}

def get_translator(lang: str, default: dict = dict(), **translations: Dict[str, str]) -> Callable[[str], str]:
    """Translator factory
    Returns a function that returns prepared translations from
    a languge dictionary for a passed key. The dictionaries have to be passed
    as kwargs, using the language as key. 
    """
    # check for the language
    if lang not in translations:
        if len(default.keys()) > 0:
            _T = default
        else:
            _T = DEFAULT
    else:
        _T = translations[lang]
    
    # define the helper
    def _get_translation(key: str):
        return _T.get(key, '')
    
    # return
    return _get_translation
