from abc import ABC, abstractmethod
import pandas as pd


class EndoscopyDrafter(ABC):
   def __init__(self, sample, pred_df, patients_df=None, procedures_df=None, polyp_df=None):
      self.sample = sample
      self.pred_df = pred_df
      # Allow patients_df and procedures_df to be optional; use empty DataFrames when not provided
      self.patients_df = patients_df if patients_df is not None else pd.DataFrame()
      self.procedures_df = procedures_df if procedures_df is not None else pd.DataFrame()
      self.polyp_df = polyp_df
      self.sample_df = self._get_sample_row(pred_df, sample)  # df row for the SINGLE sample

   def _get_sample_row(self, df, sample):
      try:
         sample_row_df = df.loc[[sample]]  # force dataframe # exact sample name match
         return sample_row_df.iloc[-1]  # if multiple, use the last one

      except KeyError:
         raise ValueError(f"Sample {sample} not found in provided dataframe.")

   @abstractmethod
   def get_indications(self):
      """Return (age, sex, indication_string)"""
      # Prefer patient CSV values, fall back to LLM-extracted sample row if patient row missing
      age = ''
      sex = ''
      try:
         patient_row = self.patients_df.loc[self.sample]
      except Exception:
         patient_row = None

      if patient_row is not None:
         age_raw = patient_row.get('age', '')
         sex = str(patient_row.get('sex', '')).lower()
      else:
         age_raw = self.sample_df.get('age', '')
         sex = str(self.sample_df.get('sex', '')).lower()

      # Safe coerce age to int
      try:
         age = int(float(age_raw)) if str(age_raw).strip() and str(age_raw).strip().lower() != 'nan' else ''
      except Exception:
         age = age_raw

      # Try procedures_df for explicit indication details
      try:
         proc_row = self.procedures_df.loc[self.sample]
      except Exception:
         proc_row = None

      if proc_row is not None:
         indications = str(proc_row.get('indication_details', '') or '')
         if indications and indications.strip().lower() != 'nan':
            return age, sex, indications

         # otherwise collect boolean indicator_* columns if present
         indications_list = []
         for col in self.procedures_df.columns:
            if col.startswith('indication_') and col != 'indication_details':
               try:
                  val = proc_row.get(col, False)
               except Exception:
                  val = False
               if val:
                  indications_list.append(col.replace('indication_', '').replace('__', '/').replace('_', ' '))
         if indications_list:
            if len(indications_list) == 1:
               return age, sex, indications_list[0]
            return age, sex, ', '.join(indications_list[:-1]) + ', and ' + indications_list[-1]

      # Final fallback: use sample_df fields if available (look for common names)
      indications = self.sample_df.get('indications') or self.sample_df.get('indication_details') or self.sample_df.get('chief_complaints') or ''
      indications = str(indications or '')
      return age, sex, indications


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
