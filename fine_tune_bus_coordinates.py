import xml.etree.ElementTree as ET
import pandas as pd


def read_graphml_coordinates(filename):
    """
    Read node coordinates from a yEd GraphML file

    Args:
        filename: Path to the GraphML file

    Returns:
        Dictionary with bus numbers as keys and (x, y) tuples as values
    """
    # Parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    # Define namespaces used in yEd GraphML files
    namespaces = {
        'g': 'http://graphml.graphdrawing.org/xmlns',
        'y': 'http://www.yworks.com/xml/graphml'
    }

    # Dictionary to store coordinates
    bus_coordinates = {}

    # Find all nodes
    for node in root.findall('.//g:node', namespaces):
        # Get node ID
        node_id = node.get('id')

        # Find the geometry element containing x, y coordinates
        geometry = node.find('.//y:Geometry', namespaces)
        if geometry is not None:
            x = float(geometry.get('x'))
            y = float(geometry.get('y'))

            # Find the label to get bus number
            label = node.find('.//y:NodeLabel', namespaces)
            if label is not None and label.text:
                bus_number = int(label.text.strip())
                bus_coordinates[bus_number] = (x, y)

    return bus_coordinates


def transform_to_uk_coordinates(bus_coordinates, uk_bounds=None):
    """
    Transform yEd pixel coordinates to UK geographical coordinates

    Args:
        bus_coordinates: Dictionary with bus numbers as keys and (x, y) tuples as values
        uk_bounds: Tuple of (lon_min, lon_max, lat_min, lat_max) for UK bounds
                  Default uses bounds that fit within windstorm contours

    Returns:
        Dictionary with bus numbers as keys and (lon, lat) tuples as values
    """
    # Default UK bounds if not specified
    if uk_bounds is None:
        # These bounds ensure we're within the windstorm contours
        uk_lon_min, uk_lon_max = -4.5, 1.0
        uk_lat_min, uk_lat_max = 50.5, 55.5
    else:
        uk_lon_min, uk_lon_max, uk_lat_min, uk_lat_max = uk_bounds

    # Extract all x and y coordinates
    x_coords = [coord[0] for coord in bus_coordinates.values()]
    y_coords = [coord[1] for coord in bus_coordinates.values()]

    # Find bounds of yEd coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    print(f"yEd coordinate bounds:")
    print(f"  X: [{x_min}, {x_max}], range: {x_max - x_min}")
    print(f"  Y: [{y_min}, {y_max}], range: {y_max - y_min}")

    print(f"\nTarget UK bounds:")
    print(f"  Longitude: [{uk_lon_min}, {uk_lon_max}]")
    print(f"  Latitude: [{uk_lat_min}, {uk_lat_max}]")

    # Transform coordinates
    transformed = {}

    for bus, (x, y) in bus_coordinates.items():
        # Linear transformation for longitude
        lon = uk_lon_min + (x - x_min) / (x_max - x_min) * (uk_lon_max - uk_lon_min)

        # Linear transformation for latitude
        # Note: Y increases downward in yEd, but latitude increases northward
        # So we need to invert the Y transformation
        lat = uk_lat_max - (y - y_min) / (y_max - y_min) * (uk_lat_max - uk_lat_min)

        # Round to 1 decimal place
        lon = round(lon, 2)
        lat = round(lat, 2)

        transformed[bus] = (lon, lat)

    return transformed


def main():
    """Main function to read GraphML and transform coordinates"""

    # Read coordinates from GraphML file
    filename = 'Input_Data/GB_Network_29bus/related_files/29_bus_GB_network_copy.graphml'
    print(f"Reading coordinates from {filename}...")

    try:
        bus_coordinates = read_graphml_coordinates(filename)
        print(f"\nSuccessfully read {len(bus_coordinates)} bus coordinates")

        # Display RAW yEd coordinates
        print("\n" + "=" * 60)
        print("RAW yEd COORDINATES (as read from GraphML):")
        print("=" * 60)

        for bus in sorted(bus_coordinates.keys()):
            x, y = bus_coordinates[bus]
            print(f"  Bus {bus:2d}: x={x:7.1f}, y={y:7.1f}")

        # Output raw coordinates in Excel format
        print("\n" + "-" * 40)
        print("Raw yEd coordinates for Excel/Python:")
        print("-" * 40)

        x_list = [bus_coordinates[i][0] for i in range(1, 30)]
        y_list = [bus_coordinates[i][1] for i in range(1, 30)]

        print("\nRaw X values:")
        print(', '.join(map(str, x_list)))

        print("\nRaw Y values:")
        print(', '.join(map(str, y_list)))

        # Output as Python dictionary
        print("\nPython dictionary format:")
        print("yed_coordinates = {")
        for bus in sorted(bus_coordinates.keys()):
            x, y = bus_coordinates[bus]
            print(f"    {bus}: {{'x': {x}, 'y': {y}}},")
        print("}")

        # Transform to UK coordinates
        print("\n" + "=" * 60)
        print("TRANSFORMING TO UK GEOGRAPHICAL COORDINATES...")
        print("=" * 60)

        uk_coordinates = transform_to_uk_coordinates(bus_coordinates)

        # Display transformed coordinates
        print("\nTransformed UK coordinates:")
        for bus in sorted(uk_coordinates.keys()):
            lon, lat = uk_coordinates[bus]
            print(f"  Bus {bus:2d}: lon={lon:5.1f}, lat={lat:4.1f}")

        # Generate Excel-ready output for transformed coordinates
        print("\n" + "=" * 60)
        print("TRANSFORMED COORDINATES FOR EXCEL UPDATE:")
        print("=" * 60)

        # Extract longitude and latitude lists in bus order
        lon_list = [uk_coordinates[i][0] for i in range(1, 30)]
        lat_list = [uk_coordinates[i][1] for i in range(1, 30)]

        print("\nGeo_lon:")
        print(', '.join(map(str, lon_list)))

        print("\nGeo_lat:")
        print(', '.join(map(str, lat_list)))

        # Save both raw and transformed coordinates to CSV
        df_raw = pd.DataFrame({
            'Bus': range(1, 30),
            'yEd_X': x_list,
            'yEd_Y': y_list
        })

        df_transformed = pd.DataFrame({
            'Bus': range(1, 30),
            'Longitude': lon_list,
            'Latitude': lat_list
        })

        # Combine into one DataFrame
        df_combined = pd.merge(df_raw, df_transformed, on='Bus')

        # Save to CSV
        df_combined.to_csv('bus_coordinates_raw_and_transformed.csv', index=False)
        print("\nAll coordinates saved to 'bus_coordinates_raw_and_transformed.csv'")

        # Also save separate files
        df_raw.to_csv('bus_coordinates_raw_yed.csv', index=False)
        df_transformed.to_csv('bus_coordinates_transformed_uk.csv', index=False)
        print("Separate files saved:")
        print("  - bus_coordinates_raw_yed.csv (raw yEd coordinates)")
        print("  - bus_coordinates_transformed_uk.csv (UK geographical coordinates)")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the GraphML file is in the current directory")


if __name__ == "__main__":
    main()