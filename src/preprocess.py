import argparse
import logging
import os
import re

import pandas as pd

logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")


def main(input_file, output_file, max_papers):
    df = pd.read_csv(input_file)

    logging.warning("done reading csv data into df")

    pat = re.compile(r"'(\S*?)'")

    df["terms_clean"] = [re.findall(pat, tt) for tt in df.terms]

    unique_terms = set(x for term in df.terms_clean for x in term)

    # for now, keep only computer vision papers
    df = df.drop(df.index[~df["terms_clean"].apply(lambda x: "cs.CV" in x)].tolist())

    # only keep certain papers
    df = df.head(max_papers)

    logging.warning(f"cleaned data, keeping only {max_papers} papers with cs.CV")

    df = df.rename(columns={"titles": "fullname", "summaries": "abstracts"})

    df = df.drop(labels=["terms", "terms_clean"], axis="columns")

    # fill this with dummy data so it doesnt break
    df["conflicts"] = "apples;bananas"

    df.to_csv(output_file, index=True, index_label="user_id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("path", type=str, help="Path to the input file to be processed")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=500,
        help="Maximum number of papers to process (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file name (default is <original name>.processed.csv)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Determine the output file name
    if args.output:
        output_file = args.output
    else:
        base_name, ext = os.path.splitext(args.path)
        output_file = f"{base_name}.processed.csv"

    # Print the arguments for debugging purposes
    logging.warning(f"processing {args.max_papers} from {args.path} into {output_file}")

    main(args.path, output_file, args.max_papers)
