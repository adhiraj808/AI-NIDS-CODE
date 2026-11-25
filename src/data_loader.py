"""
Data Loader Module for AI-NIDS.
Handles ingestion of raw CICIDS2017 CSV files, chunked loading for large captures,
automatic label column detection, and binary/multi-class label segregation.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import io
import logging
import numpy as np
import pandas as pd

from config.settings import (
    BENIGN_LABEL,
    DEFAULT_CHUNK_SIZE,
    LABEL_COLUMN,
    MALICIOUS_LABEL,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AI-NIDS.DataLoader")


class DataLoader:
    """
    Robust data loader capable of reading full or chunked CICIDS2017 network traffic datasets.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.detected_label_col: Optional[str] = None
        self.original_label_counts: Dict[str, int] = {}
        self.metadata: Dict[str, Union[str, int, float, list]] = {}

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Stripping leading/trailing whitespaces and standardizing column headers.
        In CICIDS2017, many column names contain leading spaces (e.g., ' Destination Port').
        """
        df.columns = [c.strip() for c in df.columns]
        return df

    def find_label_column(self, columns: List[str]) -> str:
        """
        Heuristically and deterministically identifies the ground truth label column.
        """
        # Exact case match
        for col in columns:
            clean_col = col.strip()
            if clean_col.lower() == "label":
                self.detected_label_col = clean_col
                return clean_col

        # Partial match if named target, class, or attack
        for col in columns:
            clean_col = col.strip()
            if clean_col.lower() in ["attack", "class", "target", "attack_type"]:
                self.detected_label_col = clean_col
                return clean_col

        raise KeyError(
            f"Could not identify a valid Label column among candidate columns: {columns[:10]}... "
            f"Expected 'Label' or equivalent."
        )

    def load_csv(
        self,
        source: Union[str, Path, io.BytesIO],
        max_rows: Optional[int] = None,
        use_chunks: bool = False,
    ) -> pd.DataFrame:
        """
        Load CSV dataset from file path or file-like buffer.

        Args:
            source: File path or IO buffer (such as Streamlit UploadedFile)
            max_rows: Optional limit of rows to load (useful for fast exploration or memory-constrained runs)
            use_chunks: Whether to read in chunks and concatenate (for very large files)

        Returns:
            pd.DataFrame: Loaded and sanitized DataFrame.
        """
        logger.info("Initiating CSV load operation...")

        try:
            if use_chunks and max_rows is None:
                chunks = []
                for chunk in pd.read_csv(
                    source,
                    chunksize=self.chunk_size,
                    low_memory=False,
                    encoding="utf-8",
                    on_bad_lines="skip",
                ):
                    chunk = self.clean_column_names(chunk)
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(
                    source,
                    nrows=max_rows,
                    low_memory=False,
                    encoding="utf-8",
                    on_bad_lines="skip",
                )
                df = self.clean_column_names(df)

        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed. Retrying with latin1 encoding...")
            df = pd.read_csv(
                source,
                nrows=max_rows,
                low_memory=False,
                encoding="latin1",
                on_bad_lines="skip",
            )
            df = self.clean_column_names(df)

        if df.empty:
            raise ValueError("Supplied dataset is completely empty.")

        label_col = self.find_label_column(df.columns.tolist())
        logger.info("Successfully loaded %d records with %d features. Label column: '%s'", len(df), len(df.columns), label_col)

        # Cache original label counts
        raw_labels = df[label_col].astype(str).str.strip()
        self.original_label_counts = raw_labels.value_counts().to_dict()

        # Update metadata
        self.metadata = {
            "total_rows": len(df),
            "total_features": len(df.columns),
            "label_column": label_col,
            "unique_original_labels": list(self.original_label_counts.keys()),
            "estimated_memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }

        return df

    def binarize_labels(
        self, df: pd.DataFrame, label_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Converts multi-class attack labels into binary classifications:
        - 'BENIGN' -> 0 (Normal traffic)
        - Everything else -> 1 ('MALICIOUS' network activity)

        Preserves original multi-class labels in 'Original_Attack_Type' column.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Cleaned DataFrame with binary Label and original labels Series.
        """
        df_out = df.copy()
        if label_col is None:
            label_col = self.detected_label_col or self.find_label_column(df_out.columns.tolist())

        raw_labels = df_out[label_col].astype(str).str.strip()

        # Preserve original attack labels
        df_out["Original_Attack_Type"] = raw_labels
        original_labels_series = raw_labels.copy()

        # Map to BENIGN / MALICIOUS
        is_benign = raw_labels.str.upper() == BENIGN_LABEL
        df_out[LABEL_COLUMN] = np.where(is_benign, BENIGN_LABEL, MALICIOUS_LABEL)

        benign_cnt = int(is_benign.sum())
        malicious_cnt = int((~is_benign).sum())

        logger.info(
            "Binarization complete: %d BENIGN (%.2f%%), %d MALICIOUS (%.2f%%)",
            benign_cnt,
            (benign_cnt / len(df_out)) * 100 if len(df_out) else 0,
            malicious_cnt,
            (malicious_cnt / len(df_out)) * 100 if len(df_out) else 0,
        )

        return df_out, original_labels_series

    def stream_chunks(
        self, filepath: Union[str, Path]
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generator yielding sanitized chunks for memory-constrained environments.
        """
        for chunk in pd.read_csv(
            filepath,
            chunksize=self.chunk_size,
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="skip",
        ):
            yield self.clean_column_names(chunk)
