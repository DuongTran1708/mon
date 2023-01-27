







# current_dir = Path(__file__).resolve().parent
# files       = list(current_dir.rglob("*.py"))
# for f in files:
#     module = f.stem
#     if module == "__init__":
#         continue
#     importlib.import_module(f"datasets.{module}")

# importlib.import_module('ltr')
# sys.modules['dlframework'] = sys.modules['ltr']
# sys.modules['dlframework.common'] = sys.modules['ltr']
# importlib.import_module('ltr.admin')
# sys.modules['dlframework.common.utils'] = sys.modules['ltr.admin']
# for m in ('model_constructor', 'stats', 'settings', 'local'):
# 	importlib.import_module('ltr.admin.' + m)
# 	sys.modules['dlframework.common.utils.' + m] = sys.modules['ltr.admin.' + m]
