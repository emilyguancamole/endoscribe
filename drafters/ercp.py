from drafters.base import EndoscopyDrafter
from docx import Document
from docx.shared import Pt
import re
from drafters.utils import add_bold_subheading


class ERCPDrafter(EndoscopyDrafter):
    def construct_recommendations(self):
        rec = []
        ercp_row = self.sample_df
        # if ercp_row.get('samples_taken', 'False') == 'True': # todo add in when reextracted
            # rec.append("Follow up pathology results.")
        rec.append("Follow up with referring provider.")
        rec.append("Finish IV fluids now.")
        rec.append("Pain control as needed.")
        return rec

    def construct_recall(self):
        pass

    def draft_doc(self):
        # Find sample, if multiple, use the first one
        
        print(f"Creating ERCP report for '{self.sample}'") 

        doc = Document()
        doc.add_heading(f'Report {self.sample}', level=1)

        doc.add_heading('Indications', level=2)
        indications = self.sample_df.get('indications', 'unknown').replace('\\n', '\n')
        doc.add_paragraph(indications)

        doc.add_heading('EGD Findings', level=2)
        paragraph = doc.add_paragraph()
        text = self.sample_df['egd_findings'].replace('\\n', '\n')
        add_bold_subheading(paragraph, text)
        doc.add_heading('ERCP Findings', level=2)
        paragraph = doc.add_paragraph()
        text = self.sample_df['ercp_findings'].replace('\\n', '\n')
        add_bold_subheading(paragraph, text)

        doc.add_heading('Impressions', level=2)
        impressions_text = self.sample_df['impressions'].strip("[]") 
        matches = re.findall(r'''(['"])(.*?)(\1)''', impressions_text) # split at commas outside quotes to get each list item
        impressions = [match[1] for match in matches]
        for i, item in enumerate(impressions, start=1):
            p = doc.add_paragraph(f"{i}. {item}")
            p.paragraph_format.space_after = Pt(0)
        return doc