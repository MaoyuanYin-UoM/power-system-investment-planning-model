"""
This script reads the half-hourly demand profile from:
    "Demand_Profile/demand_profile_2024_England_Wales_29Feb2024_removed.xlsx" (29Feb is removed to ensure 365 days)
and writes the normalized profile into: "Demand_Profile/normalized_half_hourly_demand_profile.xlsx"
and writes the extracted normalized hourly profile into:
"""


import pandas as pd

# ----------------read profile--------------

# Load the Excel file and extract the data
file_path = "Input_Data/Demand_Profile/demand_profile_2024_England_Wales_29Feb2024_removed.xlsx"
sheet_name = "demand_profile_2024"
demand_data = pd.read_excel(file_path, sheet_name=sheet_name, usecols="C", skiprows=0, nrows=17568)

# Convert to a list for easier manipulation
demand_profile = demand_data.iloc[:, 0].tolist()

# Debug: Print the first 10 values
print("Extracted Demand Profile (First 10 Values):", demand_profile[:10])
print("Length of the Demand Profile:", len(demand_profile))

# ----------------write normalized profile--------------

# Normalize the demand profile
max_demand = max(demand_profile)
normalized_profile = [value / max_demand for value in demand_profile]

# Debug: Print the first 10 normalized values
print("Normalized Profile (First 10 Values):", normalized_profile[:10])
print("Length of the Normalized Profile:", len(normalized_profile))

# Create a DataFrame to hold the normalized profile
normalized_df = pd.DataFrame(normalized_profile, columns=["Normalized_Half_Hourly_England_Wales_Demand_2024"])

# Write to a new Excel file
output_file_path = "Input_Data/Demand_Profile/normalized_half_hourly_demand_profile_year.xlsx"
normalized_df.to_excel(output_file_path, index=False)

# Debug: Confirm the file has been saved
print(f"Normalized profile saved to {output_file_path}")

# ----------------write hourly normalized profile--------------

# Create an hourly profile by averaging every two half-hourly values
hourly_normalized_profile = [
    (normalized_profile[i] + normalized_profile[i + 1]) / 2 for i in range(0, len(normalized_profile), 2)
]

# Debug: Print the first 10 hourly values
print("Hourly Normalized Profile (First 10 Values):", hourly_normalized_profile[:10])
print("Length of the Hourly Normalized Profile:", len(hourly_normalized_profile))

# Create a DataFrame to hold the hourly normalized profile
hourly_normalized_df = pd.DataFrame(hourly_normalized_profile, columns=["Normalized_Hourly_England_Wales_Demand_2024"])

# Write the hourly profile to a new Excel file
hourly_output_file_path = "Input_Data/Demand_Profile/normalized_hourly_demand_profile_year.xlsx"
hourly_normalized_df.to_excel(hourly_output_file_path, index=False)

# Debug: Confirm the file has been saved
print(f"Hourly normalized profile saved to {hourly_output_file_path}")