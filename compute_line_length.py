#!/usr/bin/env python3
"""
compute_branch_lengths.py

Reads geographic coordinates from the 'bus' sheet of the GB 29-bus network Excel file,
calculates the great-circle length of each branch, and writes the results back into a
new Excel file under the same sheets, adding the 'Length (km)' column to 'line&trafo'.
"""

import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import sys

# Hard-coded input and new output paths
INPUT_EXCEL  = "Input_Data/GB_Network_29bus/GB_Transmission_Network_29_Bus.xlsx"
OUTPUT_EXCEL = "Input_Data/GB_Network_29bus/GB_Transmission_Network_29_Bus_(length_added).xlsx"

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate great‐circle distance between two points on the Earth
    (specified in decimal degrees). Returns distance in kilometers.
    """
    R = 6371.0  # mean Earth radius [km]
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1) * cos(φ2) * sin(Δλ/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def main():
    # Load the workbook and required sheets
    xls = pd.ExcelFile(INPUT_EXCEL)
    if 'bus' not in xls.sheet_names or 'line&trafo' not in xls.sheet_names:
        sys.exit("Error: Excel file must contain sheets named 'bus' and 'line&trafo'.")

    # Read sheets into DataFrames
    bus_df    = pd.read_excel(xls, sheet_name='bus')
    branch_df = pd.read_excel(xls, sheet_name='line&trafo')

    # Build lookup: Bus ID -> (lat, lon)
    coord_lookup = {
        int(row["Bus ID"]): (row["Geo_lat"], row["Geo_lon"])
        for _, row in bus_df.iterrows()
    }

    # Compute lengths for each branch
    lengths = []
    for _, row in branch_df.iterrows():
        fb = int(row["From_bus ID"])
        tb = int(row["To_bus ID"])
        if fb not in coord_lookup or tb not in coord_lookup:
            missing = fb if fb not in coord_lookup else tb
            sys.exit(f"Error: bus {missing} missing Geo_lat/Geo_lon in 'bus' sheet.")
        lat1, lon1 = coord_lookup[fb]
        lat2, lon2 = coord_lookup[tb]
        lengths.append(haversine(lat1, lon1, lat2, lon2))

    # Assign new column
    branch_df['Length (km)'] = lengths

    # Write updated sheets to a new Excel file
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl', mode='w') as writer:
        bus_df.to_excel(writer, sheet_name='bus', index=False)
        branch_df.to_excel(writer, sheet_name='line&trafo', index=False)

    print(f"Branch lengths computed and saved in '{OUTPUT_EXCEL}'.")

if __name__ == '__main__':
    main()

