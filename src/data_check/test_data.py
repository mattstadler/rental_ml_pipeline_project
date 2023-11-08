import pandas as pd
import numpy as np
import scipy.stats
import argparse

def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #

def test_row_count(data):
    """
    Test appropriate amount of data
    """
    assert 15000 < data.shape[0] < 1000000

def test_price_range(data, min_price, max_price):
    """
    Check if all prices in the DataFrame fall within the specified range
    """
    assert data['price'].between(min_price, max_price).all(), f"Prices are not all within the range {min_price} to {max_price}"

########################################################



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--csv", 
        type=str,
        help="Input CSV file to be tested",
        required=True
    )

    parser.add_argument(
        "--ref", 
        type=str,
        help="Reference CSV file to compare the new csv to",
        required=True
    )

    parser.add_argument(
        "--kl_threshold", 
        type=str,
        help="Threshold for the KL divergence test on the neighborhood group column",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=str,
        help="Minimum accepted price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum accepted price",
        required=True
    )

    args = parser.parse_args()

    # Load data from CSV files
    data = pd.read_csv(args.csv)
    
    ref_data = pd.read_csv(args.ref)
    kl_threshold = float(args.kl_threshold)
    min_price = float(args.min_price)
    max_price = float(args.max_price)

    try:
        test_column_names(data)
        test_neighborhood_names(data)
        test_proper_boundaries(data)
        test_similar_neigh_distrib(data, ref_data, kl_threshold)
        test_row_count(data)
        test_price_range(data, min_price, max_price)
        print("All tests passed!")
    except AssertionError as e:
        print(f"A test has failed: {e}")