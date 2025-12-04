# * pep_risk/peprisc_data_simulation.py
from peprisc_bridge import PepriscBridge
import pandas as pd
# import numpy as np

#build input 
def build_input_df(
    age_years,
    gender_male_1,   # "Male" or "Female" (match what R expects)
    bmi,
    sod,
    history_of_pep,
    precut_sphincterotomy,
    minor_papilla_sphincterotomy,
    failed_cannulation,
    difficult_cannulation,
    pneumatic_dilation_of_intact_biliary_sphincter,
    pancreatic_duct_injection,
    pancreatic_duct_injections_2,
    acinarization,
    trainee_involvement,
    cholecystectomy,
    pancreo_biliary_malignancy,
    guidewire_cannulation,
    guidewire_passage_into_pancreatic_duct,
    guidewire_passage_into_pancreatic_duct_2,
    biliary_sphincterotomy,
    patient_id=1):
    """
    Returns a pandas DataFrame with 6 rows matching R's expected schema.
    """
    # repeat patient features across 6 rows
    n = 1
    df = pd.DataFrame({
        "age_years": [age_years] * n,
        "gender_male_1": [gender_male_1] * n,
        "bmi": [bmi] * n,
        "sod": [int(sod)] * n,
        "history_of_pep": [int(history_of_pep)] * n,
        "hx_of_recurrent_pancreatitis": [int(history_of_pep)] * n,  # matches your R code
        "pancreatic_sphincterotomy": [int(history_of_pep)] * n,      # matches your R code
        "precut_sphincterotomy": [int(precut_sphincterotomy)] * n,
        "minor_papilla_sphincterotomy": [int(minor_papilla_sphincterotomy)] * n,
        "failed_cannulation": [int(failed_cannulation)] * n,
        "difficult_cannulation": [int(difficult_cannulation)] * n,
        "pneumatic_dilation_of_intact_biliary_sphincter": [int(pneumatic_dilation_of_intact_biliary_sphincter)] * n,
        "pancreatic_duct_injection": [int(pancreatic_duct_injection)] * n,
        "pancreatic_duct_injections_2": [int(pancreatic_duct_injections_2)] * n,
        "acinarization": [int(acinarization)] * n,
        "trainee_involvement": [int(trainee_involvement)] * n,
        "cholecystectomy": [int(cholecystectomy)] * n,
        "pancreo_biliary_malignancy": [int(pancreo_biliary_malignancy)] * n,
        "guidewire_cannulation": [int(guidewire_cannulation)] * n,
        "guidewire_passage_into_pancreatic_duct": [int(guidewire_passage_into_pancreatic_duct)] * n,
        "guidewire_passage_into_pancreatic_duct_2": [int(guidewire_passage_into_pancreatic_duct_2)] * n,
        "biliary_sphincterotomy": [int(biliary_sphincterotomy)] * n,
        "patient_id": [patient_id] * n
    })
    print("Input DataFrame built")
    

    return df


if __name__ == "__main__":
    print("alrighty")
    # location of R bridge file
    bridge_path = "pep_risk/prediction_model/pep_risk-master/pep_risk_app/bridge_exports.R"

    #python R bridge
    pep = PepriscBridge(bridge_path,'peprisk_predict')
    print("Bridge initialized")

    input_df = build_input_df(
        age_years=55,
        gender_male_1= 1,
        bmi=28.2,
        sod=1,
        history_of_pep=0,
        precut_sphincterotomy=0,
        minor_papilla_sphincterotomy=0,
        failed_cannulation=0,
        difficult_cannulation=1,
        pneumatic_dilation_of_intact_biliary_sphincter=0,
        pancreatic_duct_injection=1,
        pancreatic_duct_injections_2=0,
        acinarization=0,
        trainee_involvement=0,
        cholecystectomy=1,
        pancreo_biliary_malignancy=0,
        guidewire_cannulation=1,
        guidewire_passage_into_pancreatic_duct=0,
        guidewire_passage_into_pancreatic_duct_2=0,
        biliary_sphincterotomy=1,
        patient_id=1
    )
    print("Input DataFrame created, calling R model")

    out = pep.peprisk_predict(input_df)

    print("Type of output:", type(out))
    print("Output of R model:\n", out)