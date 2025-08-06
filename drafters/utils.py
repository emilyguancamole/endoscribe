import re


def add_bold_text(paragraph, text):
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
