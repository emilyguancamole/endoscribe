from abc import ABC, abstractmethod
import pandas as pd

class EndoscopyDrafter(ABC):
    def __init__(self, sample, pred_df, polyp_df=None):
        self.sample = sample
        self.pred_df = pred_df
        self.polyp_df = polyp_df
        self.sample_df = self._get_sample_row(pred_df, sample) # df row for the SINGLE sample
        # print("\nsample row df\n", self.sample_df)

    def _get_sample_row(self, df, sample):
        try:
            sample_row_df = df.loc[[sample]] # force dataframe # exact sample name match
            return sample_row_df.iloc[-1] # if multiple, use the last one

        except KeyError:
            raise ValueError(f"Sample {sample} not found in provided dataframe.")
    
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

    