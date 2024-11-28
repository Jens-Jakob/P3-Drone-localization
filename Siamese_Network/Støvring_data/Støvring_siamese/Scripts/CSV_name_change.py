import csv
import os
import re

# Define the paths to your image folders
uav_folder = 'Støvring_UAV'  # Replace with the actual path
sat_folder = 'Støvring_satellite'  # Replace with the actual path

# Read the CSV file
with open('final_pairs.csv', 'r') as csvfile, open('output.csv', 'w', newline='') as outfile:
    reader = csv.reader(csvfile)
    writer = csv.writer(outfile)
    header = next(reader)
    writer.writerow(header)  # Write the header to the output file

    for row in reader:
        uav_path = row[0]
        sat_path = row[1]
        label = row[2]

        # Extract the numbers from the UAV and satellite image paths
        uav_number = re.findall(r'\d+', os.path.basename(uav_path))[-1]
        sat_number = re.findall(r'\d+', os.path.basename(sat_path))[-1]

        # Initialize variables to store the matched filenames
        uav_match_filename = None
        sat_match_filename = None

        # Search for matching UAV image filename
        for filename in os.listdir(uav_folder):
            filename_numbers = re.findall(r'\d+', filename)
            if filename_numbers:
                filename_number = filename_numbers[-1]
                if uav_number == filename_number:
                    uav_match_filename = filename  # Just the filename
                    break

        # Adjust satellite number for offset (subtract 600)
        adjusted_sat_number = str(int(sat_number) - 600)

        # Search for matching satellite image filename
        for filename in os.listdir(sat_folder):
            filename_numbers = re.findall(r'\d+', filename)
            if filename_numbers:
                filename_number = filename_numbers[-1]
                if adjusted_sat_number == filename_number:
                    sat_match_filename = filename  # Just the filename
                    break

        # Update the paths in the CSV row if matches are found
        if uav_match_filename:
            row[0] = uav_match_filename
        else:
            print(f"No matching UAV image found for number {uav_number}")
        if sat_match_filename:
            row[1] = sat_match_filename
        else:
            print(f"No matching satellite image found for number {sat_number} (adjusted to {adjusted_sat_number})")

        writer.writerow(row)  # Write the updated row to the output file
