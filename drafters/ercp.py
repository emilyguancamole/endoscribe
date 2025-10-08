from drafters.base import EndoscopyDrafter
from docx import Document
from docx.shared import Pt
import re
from drafters.utils import add_bold_subheading


class ERCPDrafter(EndoscopyDrafter):
    def construct_recommendations(self):
        rec = []
        ercp_row = self.sample_df
        if ercp_row.get('samples_taken', False): # todo reextract
            rec.append("Follow up pathology results.")
        rec.append("Finish IV fluids now.")
        rec.append("Pain control as needed.")

        # Stent recommendations
        biliary_stent = ercp_row.get('biliary_stent_type', 'None')
        if biliary_stent != "None":
            if biliary_stent=="plastic biliary 5F":
                rec.append("Repeat ERCP in 6-8 weeks for stent removal/replacement.")
            elif biliary_stent=="plastic biliary 7F":
                rec.append("Repeat ERCP in 3-4 months for stent removal/replacement.")
            elif biliary_stent=="plastic biliary 10F":
                rec.append("Repeat ERCP in ~4-5 months for stent removal/replacement.")
            elif biliary_stent=="other biliary":
                rec.append("Repeat ERCP for stent removal/replacement as clinically indicated.")
            elif biliary_stent=="FCSEMS bengign":
                rec.append("Repeat ERCP in 3 months for stent removal/replacement.")
            elif biliary_stent=="FCSEMS malignant":
                rec.append("Repeat ERCP in 6 months for stent removal/replacement.")
        if ercp_row.get('pd_stent'):
            rec.append("AXR in 2-4 weeks to confirm stent passage.")
        #todo If stent was placed and patient had indication of chronic pancreatitis/pancreatic duct strictures/pancreatic duct leaks: • Repeat ERCP for stent exchange in 3 months
            ### update when I understand if indications come from pre-written or speech

        rec.append("Follow up with referring provider.")
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
        
        doc.add_heading('Recommendations', level=2)
        recommendations = self.construct_recommendations()
        for i, item in enumerate(recommendations, start=1):
            p = doc.add_paragraph(f"{i}. {item}")
            p.paragraph_format.space_after = Pt(0)
            
        return doc