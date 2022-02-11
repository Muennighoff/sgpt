# -*- coding: utf-8 -*-
"""Applies a function to a pandas DataFrame with parallelization, error logging and progress tracking"""

import logging
import inspect
import math

from collections import namedtuple
from collections import OrderedDict
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from enum import Enum
from time import perf_counter
from typing import Any
from typing import AnyStr
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

from more_itertools import chunked
from more_itertools import flatten
import pandas as pd
from tqdm.auto import tqdm as tqdm_auto

from io_utils.plugin_io_utils import generate_unique


class ErrorHandling(Enum):
    """Enum class to identify how to handle API errors"""

    LOG = "Log"
    FAIL = "Fail"


class BatchError(ValueError):
    """Custom exception raised if the Batch function fails"""


def _parse_batch_response_default(
    batch: List[Dict], response: List[Any], output_column_names: NamedTuple
) -> List[Dict]:
    """Adds responses to each row dictionary in the batch, assuming the batch response is a list of responses
    in the same order as the batch, while keeping the existing row dictionary entries in the batch.

    Args:
        batch: Single input row from the dataframe as a dict in a list of length 1
        response: List of one or more responses returned by the API, typically a JSON string
        output_column_names: Column names to be added to the row,
            as defined in _get_unique_output_column_names

    Returns:
        batch: Same as input batch with additional columns
            corresponding to the default output columns
    """
    return [
        {
            output_column_names.response: response,
            output_column_names.error_message: "",
            output_column_names.error_type: "",
            output_column_names.error_raw: "",
            **row,
        }
        for response, row in zip(response, batch)
    ]


class DataFrameParallelizer:
    """Applies a function to a pandas DataFrame with parallelization, error logging and progress tracking.

    This class is particularly well-suited for synchronous functions calling an API, either row-by-row or by batch.

    Attributes:
        function: Any function taking a dict as input (row-by-row mode) or a list of dict (batch mode),
            and returning a response with additional information, typically a JSON string. In batch mode,
            the function is expected to return a list of responses for each row if 'DEFAULT_RESPONSE_PARSER' is used.
        error_handling: If ErrorHandling.LOG (default), log the error from the function as a warning,
            and add additional columns to the dataframe with the error message and error type.
            If ErrorHandling.FAIL, the function will fail is there is any error.
            We recommend letting the end user choose as there are contexts which justify one option or the other.
        exceptions_to_catch: Tuple of Exception classes to catch.
            Mandatory if ErrorHandling.LOG (default).
        parallel_workers: Number of concurrent threads to parallelize the function. Default is 4.
            We recommend letting the end user tune this parameter to get better performance.
        batch_support: If True, send batches of row (list of dict) to the `function`
            Else (default) send rows as dict to the function.
            This parameter should be chosen according to the nature of the function to apply.
        batch_size: Number of rows to include in each batch. Default is 10.
            Taken into account if `batch_support` is True.
            We recommend letting the end user tune this parameter if they need to increase performance.
        batch_response_parser: Function used to parse the raw response from the function in batch mode,
            and assign the actual responses and errors back to the original batch of row (list of dict).
            This is often required for batch APIs which return nested objects with a mix of responses and errors.
            This parameter is required if batch_support is True.
        output_column_prefix: Column prefix to add to the output columns of the dataframe,
            containing the `function` responses and errors. Default is "output".
            This should be overriden by the developer: if the function to apply calls an API for text translation,
            a good output_column_prefix would be "api_translation".
        verbose: If True, log raw details on any error encountered along with the error message and error type.
            Else (default) log only the error message and the error type.
            We recommend trying without verbose first. Usually, the error message is enough to diagnose the issue.
    """

    # Default number of worker threads to use in parallel - may be tuned by the end user
    DEFAULT_PARALLEL_WORKERS = 4
    # By default, we assume the function to apply is row-by-row - should be overriden in the batch case
    DEFAULT_BATCH_SUPPORT = False
    # Default number of rows in one batch - may be tuned by the end user
    DEFAULT_BATCH_SIZE = 10
    # Default response parsing function for batch_size=1 - Simply assigns the response to the response column
    DEFAULT_RESPONSE_PARSER = _parse_batch_response_default
    # Default prefix to add to output columns - should be overriden for personalized output
    DEFAULT_OUTPUT_COLUMN_PREFIX = "output"
    # Default dictionary of output column names (key) and their descriptions (value)
    OUTPUT_COLUMN_NAME_DESCRIPTIONS = OrderedDict(
        [
            ("response", "Raw response in JSON format"),
            ("error_message", "Error message"),
            ("error_type", "Error type or code"),
            ("error_raw", "Raw error"),
        ]
    )
    # By default, set verbose to False assuming error message and type are enough information in the logs
    DEFAULT_VERBOSE = False

    def __init__(
        self,
        function: Callable[[Union[Dict, List[Dict]]], Union[Dict, List[Dict]]],
        error_handling: ErrorHandling = ErrorHandling.LOG,
        exceptions_to_catch: Tuple[Exception] = (),
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
        batch_support: bool = DEFAULT_BATCH_SUPPORT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_response_parser: Callable[
            [List[Dict], Any, NamedTuple], List[Dict]
        ] = DEFAULT_RESPONSE_PARSER,
        output_column_prefix: AnyStr = DEFAULT_OUTPUT_COLUMN_PREFIX,
        verbose: bool = DEFAULT_VERBOSE,
    ):
        self.function = function
        self.error_handling = error_handling
        self.exceptions_to_catch = exceptions_to_catch
        if error_handling == ErrorHandling.LOG and not exceptions_to_catch:
            raise ValueError("Please set at least one exception in exceptions_to_catch")
        self.parallel_workers = parallel_workers
        self.batch_support = batch_support
        if not batch_support:
            batch_size = 1
        self.batch_size = batch_size
        self.batch_response_parser = batch_response_parser
        self.output_column_prefix = output_column_prefix
        self.verbose = verbose
        self._output_column_names = None  # Will be set at runtime by the run method

    def _get_unique_output_column_names(self, existing_names: List[AnyStr]) -> NamedTuple:
        """Returns a named tuple with prefixed column names and their descriptions"""
        OutputColumnNameTuple = namedtuple(
            "OutputColumnNameTuple", self.OUTPUT_COLUMN_NAME_DESCRIPTIONS.keys()
        )
        return OutputColumnNameTuple(
            *[
                generate_unique(
                    name=column_name,
                    existing_names=existing_names,
                    prefix=self.output_column_prefix,
                )
                for column_name in OutputColumnNameTuple._fields
            ]
        )

    def _apply_function_with_error_logging(
        self,
        batch: List[Dict] = None,
        **function_kwargs,
    ) -> Union[Dict, List[Dict]]:  # sourcery skip: or-if-exp-identity
        """Wraps a row-by-row or batch function with error logging
        It applies `self.function` and:
        - If batch, parse the function response to extract results and errors using `self.batch_response_parser`
            Else, in the row-by-row case, the batch only contains one row.
            We thus use the `_parse_batch_response_default` function, which simply assigns the function response
            to a new key in the dictionary, without parsing errors from the response.
            Parsing the function response to extract errors is only required for batch functions,
            as most batch APIs return succesful responses (and make users pay for the request)
            even if all rows within the batch failed from a functional perspective.
        - handles errors from the function with two methods:
            * (default) log the error message as a warning and return the row with error keys
            * fail if there is an error (if `self.error_handling == ErrorHandling.FAIL`)
        """
        output = deepcopy(batch)
        for output_column in self._output_column_names:
            for output_row in output:
                output_row[output_column] = ""
        try:
            if not self.batch_support:
                # In the row-by-row case, there is only one element in the list as batch_size=1
                response = [(self.function(row=batch[0], **function_kwargs))]
            else:
                response = self.function(batch=batch, **function_kwargs)
            output = self.batch_response_parser(
                batch=batch,
                response=response,
                output_column_names=self._output_column_names,
            )
            errors = [
                row[self._output_column_names.error_message]
                for row in output
                if row[self._output_column_names.error_message]
            ]
            if errors:
                raise BatchError(str(errors))
        except self.exceptions_to_catch + (BatchError,) as error:
            if self.error_handling == ErrorHandling.FAIL:
                raise error
            logging.warning(
                f"Function {self.function.__name__} failed on: {batch} because of error: {error}"
            )
            error_type = str(type(error).__qualname__)
            module = inspect.getmodule(error)
            if module:
                error_type = f"{module.__name__}.{error_type}"
            for output_row in output:
                output_row[self._output_column_names.error_message] = str(error)
                output_row[self._output_column_names.error_type] = error_type
                output_row[self._output_column_names.error_raw] = str(error.args)
        return output

    def _post_process_results(self, df: pd.DataFrame, results: List[Dict]) -> pd.DataFrame:
        """Combines results from the function with the input dataframe"""
        results = flatten(results)
        output_schema = {
            **{column_name: str for column_name in self._output_column_names},
            **dict(df.dtypes),
        }
        output_df = (
            pd.DataFrame.from_records(results)
            .reindex(columns=list(df.columns) + list(self._output_column_names))
            .astype(output_schema)
        )
        if not self.verbose:
            output_df.drop(labels=self._output_column_names.error_raw, axis=1, inplace=True)
        if self.error_handling == ErrorHandling.FAIL:
            error_columns = [
                self._output_column_names.error_message,
                self._output_column_names.error_type,
                self._output_column_names.error_raw,
            ]
            output_df.drop(labels=error_columns, axis=1, inplace=True, errors="ignore")
        num_error = sum(output_df[self._output_column_names.response] == "")
        num_success = len(df.index) - num_error
        logging.info(
            f"Applying function {self.function.__name__} in parallel to {len(df.index)} row(s): "
            + f"{num_success} row(s) succeeded, {num_error} failed."
        )
        return output_df

    def run(
        self,
        df: pd.DataFrame,
        **function_kwargs,
    ) -> pd.DataFrame:
        """Applies a function to a pandas.DataFrame with parallelization, error logging and progress tracking.

        The DataFrame is iterated on and fed to the function as dictionaries, row-by-row or by batches of rows.
        This process is accelerated by the use of concurrent threads and is tracked with a progress bar.
        Errors are catched if they match the `self.exceptions_to_catch` attribute and automatically logged.
        Once the whole DataFrame has been iterated on, results and errors are added as additional columns.

        Args:
            df: Input dataframe on which the function will be applied
            **function_kwargs: Arbitrary keyword arguments passed to the `function`

        Returns:
            Input dataframe with additional columns:
            - response from the `function`
            - error message if any
            - error type if any

        """
        # First, we create a generator expression to yield each row of the input dataframe.
        # Each row will be represented as a dictionary like {"column_name_1": "foo", "column_name_2": 42}
        df_row_generator = (index_series_pair[1].to_dict() for index_series_pair in df.iterrows())
        len_generator = math.ceil(len(df.index) / self.batch_size)
        logging.info(
            f"Applying function {self.function.__name__} in parallel to {len(df.index)} row(s)"
            + f" using batch size of {self.batch_size}..."
        )
        start = perf_counter()
        self._output_column_names = self._get_unique_output_column_names(existing_names=df.columns)
        pool_kwargs = function_kwargs.copy()
        for kwarg in ["function", "row", "batch"]:  # Reserved pool keyword arguments
            pool_kwargs.pop(kwarg, None)
        (futures, results) = ([], [])
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            for batch in chunked(df_row_generator, self.batch_size):
                futures.append(
                    pool.submit(
                        fn=self._apply_function_with_error_logging,
                        batch=batch,
                        **pool_kwargs,
                    )
                )
            for future in tqdm_auto(
                as_completed(futures), total=len_generator, miniters=1, mininterval=1.0
            ):
                results.append(future.result())
        output_df = self._post_process_results(df, results)
        logging.info(f"Parallelization done in {(perf_counter() - start):.2f} seconds.")
        return output_df
