def trim_empty_lines_from_csv(file_name):
    """Trims partially empty lines from a csv file which result from newline and
    other weird characters in raw data."""
    with open(file_name, 'r') as f:
        lines = f.readlines()
    with open(file_name, 'w') as f:
        for line in lines:
            if line.strip():
                f.write(line)


if __name__ == '__main__':
    trim_empty_lines_from_csv('reddit_results_sorted copy.csv')