import en_option_v3 as opt
import builtins
def mocked_input(prompt):
    return "SOFI"
builtins.input = mocked_input

try:
    opt.main()
except Exception as e:
    print(f"Exception Type: {type(e).__name__}, Message: {e}")
