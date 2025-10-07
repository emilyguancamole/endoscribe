import re

def find_terms_spacy(text: str, terms: list[str]) -> list[str]:
    ''' Return list of terms found in text using spaCy '''
    import spacy
    from negspacy.negation import Negex
    from negspacy.termsets import termset
    
    ts = termset("en_clinical")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns(), "ent_types": ["NOUN"]}, last=True)
    doc = nlp(text.lower())
    found_terms = set()
    for token in doc:
        if token.lemma_ in terms and getattr(token._, "negex", False) is False:
            found_terms.add(token.lemma_)
    return list(found_terms)

def add_bold_subheading(paragraph, text):
    """ Add bold text for specific EUS/ERCP subheadings. """

    EUS_SUBHEADINGS = ['PANCREAS', 'BILE DUCT', 'GALLBLADDER', 'LIVER', 'LYMPH NODES', 'SPLEEN', 'PERITONEUM', 'ADRENAL GLANDS', 'AMPULLA', 'AORTA AND CELIAC AXIS', 'MEDIASTINUM']
    EGD_SUBHEADINGS = ['ESOPHAGUS', 'STOMACH', 'DUODENUM']

    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.replace("ERCFindings", "ERCP Findings") # replace in original text
    all_subheadings = EUS_SUBHEADINGS + EGD_SUBHEADINGS + ["ERCP Findings"]
    pattern = r'(\b(?:' + '|'.join(map(re.escape, all_subheadings)) + r')\b:?)'
    words = re.split(pattern, text)
    for word in words:
        if word.strip().rstrip(":").upper() in all_subheadings: # Check if the word is a subheading
            run = paragraph.add_run(word.strip().upper())
            run.bold = True
            paragraph.add_run("\n")
        else:
            paragraph.add_run(word.lstrip())  # lstripping to remove leading whitespace
