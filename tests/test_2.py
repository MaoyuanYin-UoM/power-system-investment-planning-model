import pandas as pd

from data_processing.normalize_demand_profile import sheet_name

file_path = "../Input_Data/DFES_Projections/dfes_2024_main_workbook_kearsley_gsp_group.xlsx"
sheet_name = 'Maximum_Demand'

df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

print(df)