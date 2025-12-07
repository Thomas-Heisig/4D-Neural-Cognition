"""Enhanced digital sense processing module.

This module provides advanced digital data processing capabilities including:
- Natural language processing integration
- Structured data parsing
- Time-series data handling
- API data integration
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class NLPProcessor:
    """Natural language processing for text input."""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization of text into words.

        Args:
            text: Input text string.

        Returns:
            List of tokens.
        """
        # Convert to lowercase and split by whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    @staticmethod
    def text_to_vector(text: str, vocab_size: int = 256) -> np.ndarray:
        """Convert text to a simple vector representation.

        Uses character frequency as a basic vectorization method.

        Args:
            text: Input text string.
            vocab_size: Size of vocabulary (default: 256 for ASCII).

        Returns:
            Vector representation of text.
        """
        vector = np.zeros(vocab_size)

        for char in text:
            char_code = ord(char) % vocab_size
            vector[char_code] += 1

        # Normalize
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)

        return vector

    @staticmethod
    def extract_sentiment(text: str) -> float:
        """Extract simple sentiment score from text.

        Args:
            text: Input text string.

        Returns:
            Sentiment score between -1 (negative) and 1 (positive).
        """
        # Simple keyword-based sentiment analysis
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful',
                         'happy', 'love', 'best', 'fantastic', 'perfect'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst',
                         'hate', 'sad', 'poor', 'disappointing', 'fail'}

        tokens = NLPProcessor.tokenize(text)

        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / total
        return sentiment

    @staticmethod
    def extract_keywords(text: str, top_k: int = 5) -> List[Tuple[str, int]]:
        """Extract top keywords from text based on frequency.

        Args:
            text: Input text string.
            top_k: Number of top keywords to return.

        Returns:
            List of (keyword, frequency) tuples.
        """
        tokens = NLPProcessor.tokenize(text)

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                     'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
                     'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}

        filtered_tokens = [t for t in tokens if t not in stop_words]

        # Count frequencies
        freq_dict: Dict[str, int] = {}
        for token in filtered_tokens:
            freq_dict[token] = freq_dict.get(token, 0) + 1

        # Sort by frequency
        sorted_keywords = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

        return sorted_keywords[:top_k]


class StructuredDataParser:
    """Parser for structured data formats."""

    @staticmethod
    def parse_json(json_str: str) -> Dict[str, Any]:
        """Parse JSON string to dictionary.

        Args:
            json_str: JSON formatted string.

        Returns:
            Parsed dictionary.

        Raises:
            ValueError: If JSON is invalid.
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    @staticmethod
    def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary structure.

        Args:
            nested_dict: Nested dictionary to flatten.
            parent_key: Prefix for keys (used in recursion).
            sep: Separator for nested keys.

        Returns:
            Flattened dictionary.
        """
        items = []
        for k, v in nested_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(StructuredDataParser.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            StructuredDataParser.flatten_dict(item, f"{new_key}[{i}]", sep=sep).items()
                        )
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def dict_to_vector(data: Dict[str, Any], max_size: int = 100) -> np.ndarray:
        """Convert dictionary to fixed-size vector representation.

        Args:
            data: Dictionary to convert.
            max_size: Maximum size of output vector.

        Returns:
            Vector representation of dictionary.
        """
        # Flatten the dictionary
        flat_dict = StructuredDataParser.flatten_dict(data)

        # Create vector from values
        vector = []
        for key, value in flat_dict.items():
            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Use string hash as numeric representation
                vector.append(float(hash(value) % 1000) / 1000.0)

        # Pad or truncate to max_size
        if len(vector) < max_size:
            vector.extend([0.0] * (max_size - len(vector)))
        else:
            vector = vector[:max_size]

        return np.array(vector)

    @staticmethod
    def parse_csv_line(line: str, delimiter: str = ',') -> List[str]:
        """Parse CSV line into fields.

        Args:
            line: CSV line string.
            delimiter: Field delimiter.

        Returns:
            List of field values.
        """
        return [field.strip() for field in line.split(delimiter)]


class TimeSeriesProcessor:
    """Time-series data handling and analysis."""

    @staticmethod
    def normalize_timeseries(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize time-series data.

        Args:
            data: 1D numpy array of time-series data.
            method: Normalization method ('minmax' or 'zscore').

        Returns:
            Normalized time-series data.
        """
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")

        if method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val == min_val:
                return np.zeros_like(data)
            return (data - min_val) / (max_val - min_val)

        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return np.zeros_like(data)
            return (data - mean) / std

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def extract_features(data: np.ndarray, window_size: int = 10) -> Dict[str, float]:
        """Extract statistical features from time-series.

        Args:
            data: 1D numpy array of time-series data.
            window_size: Size of window for feature extraction.

        Returns:
            Dictionary of extracted features.
        """
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")

        features = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'trend': TimeSeriesProcessor._compute_trend(data),
        }

        return features

    @staticmethod
    def _compute_trend(data: np.ndarray) -> float:
        """Compute linear trend of time-series.

        Args:
            data: 1D numpy array.

        Returns:
            Trend coefficient.
        """
        n = len(data)
        if n < 2:
            return 0.0

        x = np.arange(n)
        # Linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(data)

        numerator = np.sum((x - x_mean) * (data - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return float(slope)

    @staticmethod
    def sliding_window(data: np.ndarray, window_size: int, stride: int = 1) -> List[np.ndarray]:
        """Create sliding windows over time-series.

        Args:
            data: 1D numpy array of time-series data.
            window_size: Size of each window.
            stride: Step size between windows.

        Returns:
            List of window arrays.
        """
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")

        if window_size > len(data):
            raise ValueError(f"window_size {window_size} > data length {len(data)}")

        windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            windows.append(data[i:i + window_size])

        return windows

    @staticmethod
    def detect_anomalies(data: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Detect anomalies in time-series using z-score method.

        Args:
            data: 1D numpy array of time-series data.
            threshold: Z-score threshold for anomaly detection.

        Returns:
            Boolean array indicating anomalies.
        """
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return np.zeros(len(data), dtype=bool)

        z_scores = np.abs((data - mean) / std)
        anomalies = z_scores > threshold

        return anomalies


class APIDataIntegrator:
    """Integration with external API data sources."""

    def __init__(self):
        """Initialize API data integrator."""
        self.cache: Dict[str, Any] = {}

    def process_api_response(self, response_data: Dict[str, Any]) -> np.ndarray:
        """Process API response into neural input format.

        Args:
            response_data: Dictionary containing API response.

        Returns:
            Vector representation of API data.
        """
        # Convert to structured format
        vector = StructuredDataParser.dict_to_vector(response_data)
        return vector

    def extract_numeric_data(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract numeric values from API data.

        Args:
            data: Dictionary containing API data.

        Returns:
            Array of numeric values.
        """
        numeric_values = []

        def extract_recursive(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
            elif isinstance(obj, (int, float)):
                numeric_values.append(float(obj))
            elif isinstance(obj, bool):
                numeric_values.append(1.0 if obj else 0.0)

        extract_recursive(data)
        return np.array(numeric_values) if numeric_values else np.array([0.0])

    def cache_data(self, key: str, data: Any) -> None:
        """Cache API data for later use.

        Args:
            key: Cache key.
            data: Data to cache.
        """
        self.cache[key] = data

    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached API data.

        Args:
            key: Cache key.

        Returns:
            Cached data or None if not found.
        """
        return self.cache.get(key)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()


def preprocess_digital_input(
    data: Any,
    data_type: str = 'text',
    output_size: int = 100,
) -> np.ndarray:
    """Preprocess digital input data for neural processing.

    Args:
        data: Input data (text, dict, list, or array).
        data_type: Type of input data ('text', 'json', 'timeseries', 'api').
        output_size: Desired output vector size.

    Returns:
        Preprocessed data as numpy array.
    """
    if data_type == 'text':
        if isinstance(data, str):
            vector = NLPProcessor.text_to_vector(data, vocab_size=output_size)
        else:
            raise ValueError(f"Expected string for text data, got {type(data)}")

    elif data_type == 'json':
        if isinstance(data, str):
            data = StructuredDataParser.parse_json(data)
        if isinstance(data, dict):
            vector = StructuredDataParser.dict_to_vector(data, max_size=output_size)
        else:
            raise ValueError(f"Expected dict for json data, got {type(data)}")

    elif data_type == 'timeseries':
        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            vector = TimeSeriesProcessor.normalize_timeseries(data)
            # Pad or truncate to output_size
            if len(vector) < output_size:
                vector = np.pad(vector, (0, output_size - len(vector)), mode='constant')
            else:
                vector = vector[:output_size]
        else:
            raise ValueError(f"Expected array for timeseries data, got {type(data)}")

    elif data_type == 'api':
        if isinstance(data, dict):
            integrator = APIDataIntegrator()
            vector = integrator.process_api_response(data)
            # Adjust to output_size
            if len(vector) < output_size:
                vector = np.pad(vector, (0, output_size - len(vector)), mode='constant')
            else:
                vector = vector[:output_size]
        else:
            raise ValueError(f"Expected dict for api data, got {type(data)}")

    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    return vector
