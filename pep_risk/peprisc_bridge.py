#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 10:28:44 2025

@author: sripriyankamadduluri
"""

import pandas as pd
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri, conversion

# pandas2ri.activate()
r=robjects.r
converter = conversion.get_conversion()
converter += pandas2ri.converter
    
#load R code from selected file
def load_python_r_bridge(path: str):
    # debug thing
    # note if path doesn't resolve hard code string
    # cmd = f"""
    # tryCatch(
    #   source("{path}", echo=TRUE),
    #   error=function(e) {{
    #     cat("\\n\\n===== ERROR CALL =====\\n")
    #     print(conditionCall(e))
    #     cat("\\n\\n===== ERROR MESSAGE =====\\n")
    #     print(conditionMessage(e))
    #     cat("\\n\\n===== TRACEBACK =====\\n")
    #     traceback(20)
    #     stop(e)
    #   }}
    # )
    # """
    # try:
    #     r(cmd)
    #     return True
    # except RRuntimeError:
    #     print("Stopped after printing R error details.")
    #     return False
    
    try:
        r.source(path,echo=True)
        return True
    except RRuntimeError:
        raise RuntimeError("Stopped after R error")
        return False
         
         
def load_r_peprisc_model(
    bridge_path : str, 
    fn_name: str = 'peprisk_predict'
    ):
    """establish bridge between python and r"""
    load_r_file = load_python_r_bridge(bridge_path)
    if not load_r_file:
        print("R file failed to load")
    print("file loaded successfulyy")
    if fn_name not in robjects.globalenv:
        #show available for faster debugging
        available = list(robjects.globalenv.keys())
        raise KeyError(
            f"Function '{fn_name}' not found in R globalenv.\n"
            f"Available: {available}"
            )
        
    return robjects.globalenv[fn_name]
    
    
class PepriscBridge:    

  def __init__(self,bridge_path : str, fn_name: str = "peprisk_predict"):
    self.bridge_path = bridge_path
    print(bridge_path)
    self.fn_name = fn_name
    self.peprisk_predict = load_r_peprisc_model(bridge_path,fn_name)
    
    
    
  #Invoke R peprisc model from python  
  def peprisk_predict(self, input: pd.DataFrame):

    #input: pandas DataFrame with columns expected by your R backend.
    
    #pandas -> R dataframe
    r_df = pandas2ri.DataFrame(input)
    
    #invoke peprisc_predict model from R
    
    try:
     res = self.peprisk_predict(r_df)
    except RRuntimeError:
        raise RuntimeError("R function risk_predict failed")
    
    names = list(res.names) if res.names is not None else []
    output = {}
    
    # convert o/p 
    for key in ["reference_samples", "test_paitent_prediction", "explaination_text"]:
         if key in names:
            output[key] = pandas2ri.rpy2py(res.rx2(key))

       # scalars / vectors
    for key in ["final_risk"]:
        if key in names:
               val = list(res.rx2(key))
               output[key] = val[0] if len(val) == 1 else val

    return output

        
    
    
    
    
    
      
      
    
 

         