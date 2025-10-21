import argparse
import pandas as pd
import os


def convert_checkbox_to_bool(value: str) -> bool:
    """Convert RedCap checkbox values to boolean."""
    if value=='Unchecked' or pd.isna(value) or value == '':
        return False
    return True


def clean_column_name(column_name: str) -> str:
    """Convert column names to lowercase with underscores."""
    return column_name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace(':', '').strip('_')


def process_patient_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and extract patient-related data from raw RedCap export."""
    patient_data = {}
    
    basic_columns = {
        'Participant ID': 'participant_id',
        'Eligibility': 'eligibility', 
        'MR NUMBER': 'mr_number',
        'Audio Recording ID': 'audio_recording_id',
        'Endoscopist Name': 'endoscopist_name',
        'Training Level of Endoscopist': 'endoscopist_training_level',
        'Years of experience': 'years_of_experience',
        'Date of Birth': 'dob',
        'Age': 'age',
        'Sex': 'sex',
        'Height': 'height',
        'Weight': 'weight',
        'BMI': 'bmi',
        'Race': 'race'
    }
    
    # Copy columns with name conversion
    for old_name, new_name in basic_columns.items():
        if old_name in df.columns:
            patient_data[new_name] = df[old_name]
    
    # Find columns with "Relevant Co morbidities" and convert checkbox values
    comorbidity_columns = [col for col in df.columns if 'Relevant Co morbidities' in col and 'choice=' in col]
    
    for col in comorbidity_columns:
        # Extract the condition from the column name, ex. "Relevant Co morbidities (choice=Stroke / TIA)" -> "relevant_co_morbidities_stroke_tia"
        condition = col.split('choice=')[1].replace(')', '').strip()
        new_col_name = f"relevant_co_morbidities_{clean_column_name(condition)}"
        patient_data[new_col_name] = df[col].apply(convert_checkbox_to_bool)
    if "relevant_co_morbidities_other" in df.columns:
        patient_data["relevant_co_morbidities_other"] = df['If other please elaborate']
        print("Handled other patient_data co morbidities: ", patient_data["relevant_co_morbidities_other"] )

    # # Handle variations in column names and multiple "Reason / Details" columns
    # for old_name, new_name in procedure_fields:
    #     matching_cols = [col for col in df.columns if old_name in col]
    #     if matching_cols:
    #         # Use the first matching column
    #         patient_data[new_name] = df[matching_cols[0]]
    #     elif old_name in df.columns:
    #         patient_data[new_name] = df[old_name]
    
    # #! Handle multiple "Reason / Details of prior procedure" columns -- how did they do this
    # reason_columns = [col for col in df.columns if 'Reason / Details of prior procedure' in col]
    # for i, col in enumerate(reason_columns):
    #     if i == 0:
    #         patient_data['first_previous_procedure_reason'] = df[col]
    #     elif i == 1:
    #         patient_data['second_previous_procedure_reason'] = df[col]
    #     elif i == 2:
    #         patient_data['third_previous_procedure_reason'] = df[col]
    #     elif i == 3:
    #         patient_data['fourth_previous_procedure_reason'] = df[col]
    
    return pd.DataFrame(patient_data)


def process_procedure_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and extract procedure-related data from the raw RedCap export."""
    procedure_data = {}
    
    # Basic procedure info
    basic_columns = {
        'Participant ID': 'participant_id',
        'Date of Procedure:': 'procedure_date',
        'Room Number': 'room_number',
        'Diagnosis': 'diagnosis',
        'Description of procedure': 'description_of_procedure',
        'Radiological Findings': 'radiological_findings',
        'Procedure Start Time': 'procedure_start_time',
        'Procedure End Time': 'procedure_end_time',
        'Duration of Procedure': 'duration_of_procedure'
    }
    
    for old_name, new_name in basic_columns.items():
        if old_name in df.columns:
            procedure_data[new_name] = df[old_name]
    
    # Process procedure type checkbox columns, cols have "Procedure Type (choice="
    procedure_type_columns = [col for col in df.columns if 'Procedure Type (choice=' in col]
    for col in procedure_type_columns:
        # Extract procedure type from the column name
        proc_type = col.split('choice=')[1].replace(')', '').strip()
        new_col_name = f"procedure_type_{clean_column_name(proc_type)}"
        procedure_data[new_col_name] = df[col].apply(convert_checkbox_to_bool)
    if "procedure_type_other" in df.columns:
        procedure_data["procedure_type_other"] = df["Elaborate if others"]
        print("Handled other procedure: ", procedure_data["procedure_type_other"] )
    
    # Process indication checkbox columns
    indication_columns = [col for col in df.columns if 'Indication (choice=' in col]
    for col in indication_columns:
        indication = col.split('choice=')[1].replace(')', '').strip()
        new_col_name = f"indication_{clean_column_name(indication)}"
        procedure_data[new_col_name] = df[col].apply(convert_checkbox_to_bool)
    # handle details col
    procedure_data['indication_details'] = df['Explain indication in detail']
    
    return pd.DataFrame(procedure_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--redcap_csv', type=str, required=True, help="Path to the data export (csv file) from Redcap")
    parser.add_argument('--output_dir', type=str, default="data", help="Directory to save outputs; default 'data'")
    args = parser.parse_args()
    
    if not os.path.exists(args.redcap_csv):
        raise FileNotFoundError(f"RedCap CSV file not found: {args.redcap_csv}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = pd.read_csv(args.redcap_csv)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from /RedCap")
    
    patient_df = process_patient_data(df)
    procedure_df = process_procedure_data(df)
    # todo findings_df for gold recorded findings based on procedure_type?
    
    # Export to csvs (overwrite any existing files)
    patient_csv_path = os.path.join(args.output_dir, 'patients.csv')
    procedure_csv_path = os.path.join(args.output_dir, 'procedures.csv')
    
    print(f"Saving patient data to: {patient_csv_path}, {len(patient_df.columns)} columns")
    patient_df.to_csv(patient_csv_path, index=False)
    
    print(f"Saving procedure data to: {procedure_csv_path}, {len(procedure_df.columns)} columns")
    procedure_df.to_csv(procedure_csv_path, index=False)


if __name__ == "__main__":
    """ example run: 
    python load_data.py --redcap_csv /Users/emilyguan/Downloads/EndoScribe/datasheets/irb/ENDOSCRIBEAINOTEWRIT_DATA_LABELS_2025-10-10_1347.csv """
    main()