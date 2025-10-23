from abc import ABC, abstractmethod


class EndoscopyDrafter(ABC):
   def __init__(self, sample, pred_df, patients_df, procedures_df, polyp_df=None):
       self.sample = sample
       self.pred_df = pred_df
       self.patients_df = patients_df
       self.procedures_df = procedures_df
       self.polyp_df = polyp_df
       self.sample_df = self._get_sample_row(pred_df, sample) # df row for the SINGLE sample
       # print("\nsample row df\n", self.sample_df)

   def _get_sample_row(self, df, sample):
       try:
           sample_row_df = df.loc[[sample]] # force dataframe # exact sample name match
           return sample_row_df.iloc[-1] # if multiple, use the last one


       except KeyError:
           raise ValueError(f"Sample {sample} not found in provided dataframe.")
  # 10/20 comment out for quick egd testing #todo implement method for egddrafter
   @abstractmethod
   def get_indications(self):
        """{age} year old {sex} here for an {procedure type} for {indication}"""
        age = int(self.patients_df.loc[self.sample].get('age', ''))
        sex = self.patients_df.loc[self.sample].get('sex', '').lower()
        # get indication: first find "Explain indication in detail". If none, find all True from indication_* and join with commas
        indications = str(self.procedures_df.loc[self.sample].get("indication_details", ""))
        if indications != "nan" and indications.strip() != "": 
            return age, sex, indications
        indications = []
        for col in self.procedures_df.columns:
            if col.startswith('indication_') and self.procedures_df.loc[self.sample].get(col, False) and col != 'indication_details':
                indications.append(col.replace('indication_', '').replace('__', '/').replace('_', ' '))
        # join indications with commas and "and" before last one
        if len(indications) == 0:
            indication_str = ""
        elif len(indications) == 1:
            indication_str = indications[0]
        else:
            indication_str = ', '.join(indications[:-1]) + ', and ' + indications[-1]
        return age, sex, indication_str


   @abstractmethod
   def construct_recall(self):
       pass


   @abstractmethod
   def construct_recommendations(self):
       # Force subclasses to implement
       pass


   @abstractmethod
   def draft_doc(self):
       # Main entry: produces the final report
       pass
