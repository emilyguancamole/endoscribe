import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv()
    
# # clean up data
#* moved to llama_vllm.py
# def convert_fields_to_int(df):
#     """ Convert data fields from colonoscopies and polyps tables to int """
#     cols_to_convert = [
#         'bbps_simple', 'bbps_right', 'bbps_transverse', 'bbps_left', 'bbps_total', 'polyp_count', # colonoscopies
#         'nice_class', # polyps
#     ]
#     for col in cols_to_convert:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
#         else:
#             print(f"Column '{col}' not found in DataFrame.")
#     return df

# def convert_fields_to_float(df):
#     """ Convert the fields to float from polyps table """
#     cols_to_convert = [
#         'size_min_mm', 'size_max_mm', # polyps
#     ]
#     for col in cols_to_convert:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
#         else:
#             print(f"Column '{col}' not found in DataFrame.")
#     return df

## Read in data -- this should be already NORMALIZED data extracted by LLM
# db_csv = 'extracted/colonoscopy/102024/abstract_colonoscopies.csv'
# db_csv = 'extracted/colonoscopy/102024/abstract_polyps.csv'
# df = pd.read_csv(db_csv)

## Data conversions
# df = convert_fields_to_int(df)
#! only for colonoscopies table
# df["scope_in_time"] = pd.to_datetime(df["scope_in_time"]).dt.time # should already be stored in time format in the csv
# df["scope_out_time"] = pd.to_datetime(df["scope_out_time"]).dt.time
#! only need for polyps table
# df = convert_fields_to_float(df) 
# print("Data types after conversion:", df.dtypes)


def write_predictions_to_postgres(col_df, polyp_df):
    """Write colonoscopy data and polyp extraction data to Postgres tables"""
    # connection settings
    user = 'postgres'
    pw = os.getenv('POSTGRES_PASSWORD')
    host = 'localhost'
    port = '5432'
    db = 'endo'
    # table_name = 'abstract_polyps' #! 'abstract_colonoscopies' #! 'abstract_polyps'

    col_df = col_df.reset_index()
    polyp_df = polyp_df.reset_index()

    engine = create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}") # create connection engine

    try:
        col_df.to_sql("abstract_colonoscopies", engine, if_exists='replace', index=False) # upload data #* if_exists can also ="append"
        polyp_df.to_sql("abstract_polyps", engine, if_exists='replace', index=False)
        print(f"Successfully uploaded colonoscopy and polyp data to Postgres")
    except Exception as e:
        print("Failed to write to Postgres:", e)

if __name__ == '__main__':

    print("Converting")