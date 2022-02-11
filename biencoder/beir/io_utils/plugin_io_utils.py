# -*- coding: utf-8 -*-
"""Module with read/write utility functions which are *not* based on the Dataiku API"""

import re
import logging
import functools
from typing import List, AnyStr, Union, Callable
from time import perf_counter

import pandas as pd
import numpy as np


def clean_empty_list(sequence: List) -> Union[List, AnyStr]:
    """If the input sequence is a valid non-empty list, return list, else an empty string

    Args:
        sequence: Original list

    Returns:
       Original list or empty string

    """
    output = ""
    if isinstance(sequence, list):
        if len(sequence) != 0:
            output = sequence
    return output


def unique_list(sequence: List) -> List:
    """Make a list unique, ordering values by order of appearance in the original list

    Args:
        sequence: Original list

    Returns:
       List with unique elements ordered by appearance in the original list

    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def truncate_text_list(text_list: List[AnyStr], num_characters: int = 140) -> List[AnyStr]:
    """Truncate a list of strings to a given number of characters

    Args:
        text_list: List of strings
        num_characters: Number of characters to truncate each string to

    Returns:
       List with truncated strings

    """
    output_text_list = []
    for text in text_list:
        if len(text) > num_characters:
            output_text_list.append(text[:num_characters] + " (...)")
        else:
            output_text_list.append(text)
    return output_text_list


def clean_text_df(df: pd.DataFrame, dropna_columns: List[AnyStr] = None) -> pd.DataFrame:
    """Clean a pandas.DataFrame with text columns to remove empty strings and NaNs values in the dataframe

    Args:
        df: Input pandas.DataFrame which should contain only text
        dropna_columns: Optional list of column names where empty strings and NaN should be checked
            Default is None, which means that all columns will be checked

    Returns:
       pandas.DataFrame with rows dropped in case of empty strings or NaN values

    """
    for col in df.columns:
        df[col] = df[col].str.strip().replace("", np.NaN)
    df = df.dropna(subset=dropna_columns)
    return df


def generate_unique(name: AnyStr, existing_names: List[AnyStr], prefix: AnyStr = None) -> AnyStr:
    """Generate a unique name among existing ones by suffixing a number and adding a prefix

    Args:
        name: Input name
        existing_names: List of existing names
        prefix: Optional prefix to add to the output name

    Returns:
       Unique name with a number suffix in case of conflict, and an optional prefix

    """
    name = re.sub(r"[^\x00-\x7F]", "_", name).replace(
        " ", "_"
    )  # replace non ASCII and whitespace characters by an underscore _
    if prefix:
        new_name = f"{prefix}_{name}"
    else:
        new_name = name
    for j in range(1, 1001):
        if new_name not in existing_names:
            return new_name
        new_name = f"{new_name}_{j}"
    raise RuntimeError(f"Failed to generated a unique name for '{name}'")


def move_columns_after(df: pd.DataFrame, columns_to_move: List[AnyStr], after_column: AnyStr) -> pd.DataFrame:
    """Reorder columns by moving a list of columns after another column

    Args:
        df: Input pandas.DataFrame
        columns_to_move: List of column names to move
        after_column: Name of the columns to move columns after

    Returns:
       pandas.DataFrame with reordered columns

    """
    after_column_position = df.columns.get_loc(after_column) + 1
    reordered_columns = (
        df.columns[:after_column_position].tolist() + columns_to_move + df.columns[after_column_position:].tolist()
    )
    df.reindex(columns=reordered_columns)
    return df


def time_logging(log_message: AnyStr):
    """Decorator to log timing with a custom message"""

    def inner_function(function: Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            logging.info(log_message + "...")
            value = function(*args, **kwargs)
            end = perf_counter()
            logging.info(log_message + f": done in {end - start:.2f} seconds")
            return value

        return wrapper

    return inner_function
