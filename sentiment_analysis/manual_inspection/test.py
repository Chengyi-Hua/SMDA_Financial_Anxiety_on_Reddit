from pathlib import Path
import csv

manual_path = Path(
    r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\manual_inspection\manual_inspection_blinded.csv"
)

validation_path = Path(
    r"D:\Users\cheng\Documents\GitHub\SMDA_Financial_Anxiety_on_Reddit\sentiment_analysis\manual_inspection\validation_result.csv"
)

output_path = manual_path.with_name("manual_inspection_blinded_filled.csv")


def csv_escape(value):
    """
    Minimal CSV escaping for values we insert.
    """
    if value is None:
        value = ""
    value = str(value)

    if any(char in value for char in [",", '"', "\n", "\r"]):
        value = '"' + value.replace('"', '""') + '"'

    return value


# ------------------------------------------------------------
# Step 1: Build lookup from validation_result.csv
# We only read the first 3 columns:
# inspection_id, manual_text_sentiment, manual_emoji_function
# ------------------------------------------------------------

updates = {}

with open(validation_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)

    header = next(reader)

    id_idx = header.index("inspection_id")
    sentiment_idx = header.index("manual_text_sentiment")
    emoji_idx = header.index("manual_emoji_function")

    for row in reader:
        if not row:
            continue

        inspection_id = row[id_idx]

        updates[inspection_id] = {
            "manual_text_sentiment": row[sentiment_idx],
            "manual_emoji_function": row[emoji_idx],
        }


# ------------------------------------------------------------
# Step 2: Process manual_inspection_blinded.csv line by line
# We split only the first 5 commas:
#
# inspection_id,
# manual_text_sentiment,
# manual_text_sentiment_notes,
# manual_emoji_function,
# manual_emoji_notes,
# rest_of_row
#
# The long Reddit text stays untouched.
# ------------------------------------------------------------

filled_count = 0
missing_count = 0

with open(manual_path, "r", encoding="utf-8-sig", newline="") as infile, \
     open(output_path, "w", encoding="utf-8-sig", newline="") as outfile:

    # Write header unchanged
    header_line = infile.readline()
    outfile.write(header_line)

    for line_number, line in enumerate(infile, start=2):
        newline = "\n" if line.endswith("\n") else ""
        line_no_newline = line.rstrip("\r\n")

        # Split only the first 5 commas, preserving the rest exactly
        parts = line_no_newline.split(",", 5)

        if len(parts) < 6:
            print(f"Skipping malformed line {line_number}: not enough columns")
            outfile.write(line)
            continue

        inspection_id = parts[0]

        old_text_sentiment = parts[1]
        old_text_sentiment_notes = parts[2]
        old_emoji_function = parts[3]
        old_emoji_notes = parts[4]
        rest_of_row = parts[5]

        if inspection_id in updates:
            new_text_sentiment = updates[inspection_id]["manual_text_sentiment"]
            new_emoji_function = updates[inspection_id]["manual_emoji_function"]

            filled_count += 1
        else:
            new_text_sentiment = old_text_sentiment
            new_emoji_function = old_emoji_function

            missing_count += 1

        new_first_columns = [
            inspection_id,
            new_text_sentiment,
            old_text_sentiment_notes,
            new_emoji_function,
            old_emoji_notes,
        ]

        new_line = ",".join(csv_escape(x) for x in new_first_columns)
        new_line = new_line + "," + rest_of_row + newline

        outfile.write(new_line)


print(f"Saved filled file to: {output_path}")
print(f"Rows updated: {filled_count}")
print(f"Rows without validation match: {missing_count}")