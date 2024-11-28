import json

# Load the JSON file
file_path = '/Users/jens-jakobskotingerslev/Desktop/Study abroad/top_50_matches_dis.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to remove "patch" from filenames
def remove_patch_from_filenames(data):
    updated_data = {}
    for key, value in data.items():
        updated_value = [filename.replace("_patch_", "_") for filename in value]
        updated_data[key] = updated_value
    return updated_data

# Apply the function to remove "patch"
updated_data = remove_patch_from_filenames(data)

# Save the updated JSON file
updated_file_path = 'rizzy_dizzy_fitty.json'
with open(updated_file_path, 'w') as updated_file:
    json.dump(updated_data, updated_file, indent=4)

updated_file_path
