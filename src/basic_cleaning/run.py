#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
from wandb_utils.log_artifact import log_artifact
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #

    logger.info(f"Fetching data {args.input_artifact}")
    local_path = wandb.use_artifact("sample.csv:latest").file()
    df = pd.read_csv(local_path)

    # Drop outliers
    min_price = 10
    max_price = 350
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save df to csv
    df.to_csv("clean_sample.csv", index=False)

    # Log artifact to wandb
    logger.info(f"Uploading {args.output_artifact} to Weights & Biases")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    ######################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the artifact to clean",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the cleaned output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="A brief desciption of this artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="The minimun price of a rental to consider",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="The maximum price of a rental to consider",
        required=True
    )


    args = parser.parse_args()

    go(args)