import csv
level={
    'Spike': '2',
}

input_file = 'simple_2_modified.csv'
output_file = 'simple_2_hh_modified.csv'


# Function to replace keys with values in a given string
def replace_keys(text):
    for key, value in level.items():
        text = text.replace(key, value)
    return text


# Read the input CSV file, replace keys, and write to the output CSV file
with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        modified_row = [replace_keys(cell) for cell in row]
        writer.writerow(modified_row)

print("Replacement completed. Check the 'file_modified.csv' for the updated content.")

