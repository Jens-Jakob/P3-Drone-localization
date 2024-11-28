import json
import sys
import argparse


def extract_reference_image_names(input_file, output_file):
    try:
        # Read the original JSON data
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Process the data to keep only "reference_image_name"
        modified_data = {}
        for image, references in data.items():
            modified_data[image] = [ref['reference_image_name'] for ref in references]

        # Write the modified data to the output file
        with open(output_file, 'w') as f:
            json.dump(modified_data, f, indent=4)

        print(f"Successfully wrote the modified data to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file}' is not valid JSON.")
    except KeyError as e:
        print(f"Error: Missing expected key {e} in the JSON data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract 'reference_image_name' from a JSON file.")
    parser.add_argument('input_file', help='Path to the input JSON file.')
    parser.add_argument('output_file', help='Path to the output JSON file.')

    args = parser.parse_args()

    extract_reference_image_names(args.input_file, args.output_file)

if __name__ == "__main__":
    main()

