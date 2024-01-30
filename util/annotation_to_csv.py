import csv
import os


# Example usage
# txt_to_csv('path_to_your_txt_file.txt', 'path_to_output_csv_file.csv')


def txt_to_csv(txt_file_path, csv_file_path):
    with open(txt_file_path, 'r') as txt_file, open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Time', 'Sample #', 'Type'])  # Write headers

        for i, line in enumerate(txt_file):
            if i == 0:
                continue

            parts = line.split()
            if len(parts) >= 3:
                # Extract the desired columns: Time, Sample #, and Type
                time, sample, type_ = parts[0], parts[1], parts[2]
                writer.writerow([time, sample, type_])


if __name__ == '__main__':
    directory = 'data'

    # Iterate over files in the directory
    for filename in os.listdir('data'):
        if filename.endswith('annotations.txt'):
            txt_file_path = os.path.join(directory, filename)
            csv_file_path = os.path.join(directory, filename.replace('annotations.txt', 'annotations.csv'))
            txt_to_csv(txt_file_path, csv_file_path)
            print(f"Converted {filename} to CSV.")
