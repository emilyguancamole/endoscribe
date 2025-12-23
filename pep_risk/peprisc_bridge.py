#* pep_risk/peprisc_bridge.py
import os
import pandas as pd
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

r = robjects.r
    
def load_python_r_bridge(path: str):
    """load R code from selected file"""
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
        # Set R working directory to project root. should be .../pep_risk-master/pep_risk_app/bridge_exports.R
        abs_path = os.path.abspath(path)
        project_root = os.path.dirname(os.path.dirname(abs_path))
        try:
            r.setwd(project_root)
        except Exception:
            pass
        r.source(abs_path, echo=True)
        return True
    except RRuntimeError:
        raise RuntimeError("Stopped after R error")
         
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
    self._r_function = load_r_peprisc_model(bridge_path,fn_name)
    
  # Invoke R peprisc model from python  
  def peprisk_predict(self, input: pd.DataFrame):    
    # Use local converter context to handle conversions safely
    with localconverter(pandas2ri.converter):
        # pandas -> R dataframe
        r_df = pandas2ri.py2rpy(input)
        # invoke peprisc_predict model from R
        try:
            res = self._r_function(r_df)
            print("R function executed successfully")
        except RRuntimeError:
            raise RuntimeError("R function risk_predict failed")

        output = {}
        # res is a Python mapping (e.g. OrderedDict) or pandas objects
        if isinstance(res, dict) or hasattr(res, 'items'):
            for key, val in res.items():
                # if val is an rpy2 object that needs conversion
                try:
                    if getattr(val, 'rclass', None) is not None or hasattr(val, 'rx2'):
                        output[key] = pandas2ri.rpy2py(val)
                    else:
                        output[key] = val
                except Exception:
                    output[key] = val
        else:
            # Fallback: try to convert the whole object
            try:
                converted = pandas2ri.rpy2py(res)
                if isinstance(converted, dict):
                    output = converted
                else:
                    output = {"result": converted}
            except Exception:
                output = {"result": res}

    return output

        
    
    
    
    
    
      
      
    
 

         