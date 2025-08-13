import pandas as pd
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
# Navigate to the project root
project_root = script_dir.parent

file_path = project_root / "Input_Data" / "bus_specific_hourly_demand_profiles_Kearsley_GSP_group_2024_final.xlsx"
sheet_name = 'BSP MW'

df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

print(df)