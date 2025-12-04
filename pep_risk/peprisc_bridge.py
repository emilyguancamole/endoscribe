#* pep_risk/peprisc_bridge.py
import pandas as pd
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri, conversion
pandas2ri.activate()
r=robjects.r
converter = conversion.get_conversion()
converter += pandas2ri.converter
    
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
        r.source(path,echo=True)
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
    
    
    
  #Invoke R peprisc model from python  
  def peprisk_predict(self, input: pd.DataFrame):

    #input: pandas DataFrame with columns expected by your R backend.
    
    # Use local converter context to handle threading issues
    from rpy2.robjects.conversion import localconverter
    
    with localconverter(pandas2ri.converter):
        #pandas -> R dataframe
        r_df = pandas2ri.py2rpy(input)

        # invoke peprisc_predict model from R
        try:
            res = self._r_function(r_df)
        except RRuntimeError:
            raise RuntimeError("R function risk_predict failed")

        output = {}

        # If res is an rpy2 object with .names / rx2, extract like before
        if getattr(res, 'names', None) is not None:
            names = list(res.names)
            for key in names:
                try:
                    val = res.rx2(key)
                    output[key] = pandas2ri.rpy2py(val)
                except Exception:
                    # fallback: assign raw value
                    try:
                        output[key] = pandas2ri.rpy2py(res.rx2(key))
                    except Exception:
                        output[key] = res.rx2(key)

        else:
            # res may already be a Python mapping (e.g., OrderedDict) or pandas objects
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
                # last resort: try to convert the whole object
                try:
                    converted = pandas2ri.rpy2py(res)
                    if isinstance(converted, dict):
                        output = converted
                    else:
                        output = {"result": converted}
                except Exception:
                    output = {"result": res}

    return output

        
    
    
    
    
    
      
      
    
 

         