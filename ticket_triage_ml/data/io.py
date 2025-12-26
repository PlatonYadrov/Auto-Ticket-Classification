"""IO utilities for reading and writing data files."""

from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger


def read_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Read data from CSV or Parquet file.

    Args:
        file_path: Path to the data file. Supports .csv and .parquet extensions.

    Returns:
        DataFrame containing the loaded data.

    Raises:
        ValueError: If file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        logger.debug(f"Reading CSV file: {file_path}")
        return pd.read_csv(file_path)
    elif suffix == ".parquet":
        logger.debug(f"Reading Parquet file: {file_path}")
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .parquet")


def write_data(
    dataframe: pd.DataFrame,
    file_path: Union[str, Path],
    index: bool = False,
) -> None:
    """Write DataFrame to CSV or Parquet file.

    Args:
        dataframe: DataFrame to save.
        file_path: Path to save the file. Extension determines format.
        index: Whether to include the index in the output file.

    Raises:
        ValueError: If file extension is not supported.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        logger.debug(f"Writing CSV file: {file_path}")
        dataframe.to_csv(file_path, index=index)
    elif suffix == ".parquet":
        logger.debug(f"Writing Parquet file: {file_path}")
        dataframe.to_parquet(file_path, index=index)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .parquet")

    logger.info(f"Saved {len(dataframe)} records to {file_path}")
