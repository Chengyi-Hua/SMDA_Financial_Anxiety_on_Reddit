import csv
from datetime import datetime, timezone
from pathlib import Path

import requests
from utils.constants import *

def to_epoch(year, month=1, day=1):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())

def fetch_post_count(subreddit, year):
    after = to_epoch(year, 1, 1)
    before = to_epoch(year + 1, 1, 1)

    params = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "limit": "auto",
        "sort": "asc",
    }

    total_count = 0
    current_after = after

    while True:
        params["after"] = current_after
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()

        posts = response.json().get("data", [])
        if not posts:
            break

        total_count += len(posts)
        current_after = posts[-1]["created_utc"]

        if len(posts) < 1000:
            break

    return total_count

def create_post_counts_csv():
    rows = []

    for subreddit in SUBREDDITS:
        for year in YEARS:
            try:
                count = fetch_post_count(subreddit, year)
                rows.append({"subreddit": subreddit, "year": year, "count": count})
                print(f"{subreddit} {year}: {count}")
            except Exception as exc:
                rows.append({"subreddit": subreddit, "year": year, "count": None})
                print(f"{subreddit} {year}: ERROR - {exc}")

    output_path = Path(__file__).resolve().parent.parent / "post_counts.csv"

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["subreddit", "year", "count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {output_path}")


def main():
    create_post_counts_csv()
