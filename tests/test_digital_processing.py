"""Tests for digital_processing module."""

import pytest
import numpy as np
from src.digital_processing import (
    NLPProcessor,
    StructuredDataParser,
    TimeSeriesProcessor,
    APIDataIntegrator,
    preprocess_digital_input,
)


class TestNLPProcessor:
    """Test NLP processing functionality."""

    def test_tokenize_simple(self):
        """Test basic tokenization."""
        text = "Hello world, this is a test!"
        tokens = NLPProcessor.tokenize(text)
        assert tokens == ['hello', 'world', 'this', 'is', 'a', 'test']

    def test_tokenize_empty(self):
        """Test tokenization of empty string."""
        tokens = NLPProcessor.tokenize("")
        assert tokens == []

    def test_tokenize_punctuation(self):
        """Test tokenization with heavy punctuation."""
        text = "Hello! How are you? I'm fine."
        tokens = NLPProcessor.tokenize(text)
        assert 'hello' in tokens
        assert 'how' in tokens
        assert 'm' in tokens  # "I'm" becomes "I" and "m"

    def test_text_to_vector_length(self):
        """Test text vectorization output length."""
        text = "test"
        vector = NLPProcessor.text_to_vector(text, vocab_size=256)
        assert len(vector) == 256

    def test_text_to_vector_normalized(self):
        """Test that vector is normalized."""
        text = "hello"
        vector = NLPProcessor.text_to_vector(text)
        assert np.isclose(np.sum(vector), 1.0)

    def test_text_to_vector_empty(self):
        """Test vectorization of empty text."""
        vector = NLPProcessor.text_to_vector("", vocab_size=256)
        assert len(vector) == 256
        assert np.sum(vector) == 0.0

    def test_extract_sentiment_positive(self):
        """Test sentiment extraction for positive text."""
        text = "This is great and amazing!"
        sentiment = NLPProcessor.extract_sentiment(text)
        assert sentiment > 0

    def test_extract_sentiment_negative(self):
        """Test sentiment extraction for negative text."""
        text = "This is terrible and awful!"
        sentiment = NLPProcessor.extract_sentiment(text)
        assert sentiment < 0

    def test_extract_sentiment_neutral(self):
        """Test sentiment extraction for neutral text."""
        text = "The cat sat on the mat"
        sentiment = NLPProcessor.extract_sentiment(text)
        assert sentiment == 0.0

    def test_extract_sentiment_mixed(self):
        """Test sentiment with mixed emotions."""
        text = "This is good but also bad"
        sentiment = NLPProcessor.extract_sentiment(text)
        assert -1 <= sentiment <= 1

    def test_extract_keywords_basic(self):
        """Test keyword extraction."""
        text = "python python python is great great"
        keywords = NLPProcessor.extract_keywords(text, top_k=2)
        assert len(keywords) == 2
        assert keywords[0][0] == 'python'
        assert keywords[0][1] == 3

    def test_extract_keywords_with_stopwords(self):
        """Test that stop words are filtered."""
        text = "the the the cat is on the mat"
        keywords = NLPProcessor.extract_keywords(text, top_k=5)
        # 'the', 'is', 'on' should be filtered
        keyword_words = [kw[0] for kw in keywords]
        assert 'the' not in keyword_words
        assert 'is' not in keyword_words

    def test_extract_keywords_empty(self):
        """Test keyword extraction from empty text."""
        keywords = NLPProcessor.extract_keywords("", top_k=5)
        assert len(keywords) == 0


class TestStructuredDataParser:
    """Test structured data parsing."""

    def test_parse_json_valid(self):
        """Test parsing valid JSON."""
        json_str = '{"name": "test", "value": 42}'
        result = StructuredDataParser.parse_json(json_str)
        assert result['name'] == 'test'
        assert result['value'] == 42

    def test_parse_json_invalid(self):
        """Test parsing invalid JSON."""
        json_str = '{"name": "test", invalid}'
        with pytest.raises(ValueError, match="Invalid JSON"):
            StructuredDataParser.parse_json(json_str)

    def test_flatten_dict_simple(self):
        """Test flattening simple nested dict."""
        nested = {'a': 1, 'b': {'c': 2, 'd': 3}}
        flat = StructuredDataParser.flatten_dict(nested)
        assert flat['a'] == 1
        assert flat['b.c'] == 2
        assert flat['b.d'] == 3

    def test_flatten_dict_with_list(self):
        """Test flattening dict with list."""
        nested = {'items': [1, 2, 3]}
        flat = StructuredDataParser.flatten_dict(nested)
        assert flat['items[0]'] == 1
        assert flat['items[1]'] == 2
        assert flat['items[2]'] == 3

    def test_flatten_dict_nested_list(self):
        """Test flattening dict with nested list of dicts."""
        nested = {'users': [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]}
        flat = StructuredDataParser.flatten_dict(nested)
        assert flat['users[0].name'] == 'Alice'
        assert flat['users[0].age'] == 30
        assert flat['users[1].name'] == 'Bob'

    def test_flatten_dict_empty(self):
        """Test flattening empty dict."""
        flat = StructuredDataParser.flatten_dict({})
        assert flat == {}

    def test_dict_to_vector_length(self):
        """Test vector length from dict."""
        data = {'a': 1, 'b': 2, 'c': 3}
        vector = StructuredDataParser.dict_to_vector(data, max_size=100)
        assert len(vector) == 100

    def test_dict_to_vector_numeric(self):
        """Test vector from numeric data."""
        data = {'x': 10, 'y': 20.5}
        vector = StructuredDataParser.dict_to_vector(data, max_size=10)
        assert vector[0] == 10.0
        assert vector[1] == 20.5

    def test_dict_to_vector_bool(self):
        """Test vector from boolean data."""
        data = {'flag': True, 'switch': False}
        vector = StructuredDataParser.dict_to_vector(data, max_size=10)
        assert vector[0] == 1.0
        assert vector[1] == 0.0

    def test_dict_to_vector_string(self):
        """Test vector from string data."""
        data = {'name': 'test'}
        vector = StructuredDataParser.dict_to_vector(data, max_size=10)
        # String should be hashed to numeric
        assert 0.0 <= vector[0] <= 1.0

    def test_dict_to_vector_padding(self):
        """Test vector padding."""
        data = {'a': 1}
        vector = StructuredDataParser.dict_to_vector(data, max_size=10)
        # Should pad with zeros
        assert vector[-1] == 0.0

    def test_dict_to_vector_truncation(self):
        """Test vector truncation."""
        data = {f'key{i}': i for i in range(200)}
        vector = StructuredDataParser.dict_to_vector(data, max_size=50)
        assert len(vector) == 50

    def test_parse_csv_line_simple(self):
        """Test CSV line parsing."""
        line = "name,age,city"
        fields = StructuredDataParser.parse_csv_line(line)
        assert fields == ['name', 'age', 'city']

    def test_parse_csv_line_custom_delimiter(self):
        """Test CSV with custom delimiter."""
        line = "name;age;city"
        fields = StructuredDataParser.parse_csv_line(line, delimiter=';')
        assert fields == ['name', 'age', 'city']

    def test_parse_csv_line_with_spaces(self):
        """Test CSV with spaces."""
        line = " name , age , city "
        fields = StructuredDataParser.parse_csv_line(line)
        assert fields == ['name', 'age', 'city']


class TestTimeSeriesProcessor:
    """Test time-series processing."""

    def test_normalize_timeseries_minmax(self):
        """Test min-max normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = TimeSeriesProcessor.normalize_timeseries(data, method='minmax')
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0

    def test_normalize_timeseries_zscore(self):
        """Test z-score normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = TimeSeriesProcessor.normalize_timeseries(data, method='zscore')
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-10)

    def test_normalize_timeseries_constant(self):
        """Test normalization of constant series."""
        data = np.array([5.0, 5.0, 5.0])
        normalized = TimeSeriesProcessor.normalize_timeseries(data, method='minmax')
        assert np.all(normalized == 0.0)

    def test_normalize_timeseries_invalid_method(self):
        """Test invalid normalization method."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown normalization method"):
            TimeSeriesProcessor.normalize_timeseries(data, method='invalid')

    def test_normalize_timeseries_wrong_dimension(self):
        """Test normalization with wrong dimensions."""
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Expected 1D array"):
            TimeSeriesProcessor.normalize_timeseries(data)

    def test_extract_features_basic(self):
        """Test feature extraction."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        features = TimeSeriesProcessor.extract_features(data)
        assert 'mean' in features
        assert 'std' in features
        assert 'min' in features
        assert 'max' in features
        assert features['mean'] == 3.0
        assert features['min'] == 1.0
        assert features['max'] == 5.0

    def test_extract_features_trend(self):
        """Test trend computation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        features = TimeSeriesProcessor.extract_features(data)
        # Should have positive trend
        assert features['trend'] > 0

    def test_extract_features_wrong_dimension(self):
        """Test feature extraction with wrong dimensions."""
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Expected 1D array"):
            TimeSeriesProcessor.extract_features(data)

    def test_sliding_window_basic(self):
        """Test sliding window."""
        data = np.array([1, 2, 3, 4, 5])
        windows = TimeSeriesProcessor.sliding_window(data, window_size=3, stride=1)
        assert len(windows) == 3
        assert np.array_equal(windows[0], np.array([1, 2, 3]))
        assert np.array_equal(windows[1], np.array([2, 3, 4]))

    def test_sliding_window_with_stride(self):
        """Test sliding window with stride."""
        data = np.array([1, 2, 3, 4, 5, 6])
        windows = TimeSeriesProcessor.sliding_window(data, window_size=2, stride=2)
        assert len(windows) == 3

    def test_sliding_window_too_large(self):
        """Test sliding window larger than data."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="window_size"):
            TimeSeriesProcessor.sliding_window(data, window_size=5)

    def test_sliding_window_wrong_dimension(self):
        """Test sliding window with wrong dimensions."""
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Expected 1D array"):
            TimeSeriesProcessor.sliding_window(data, window_size=2)

    def test_detect_anomalies_basic(self):
        """Test anomaly detection."""
        data = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0])
        anomalies = TimeSeriesProcessor.detect_anomalies(data, threshold=2.0)
        assert anomalies[3] == True  # 10.0 is anomaly
        assert anomalies[0] == False

    def test_detect_anomalies_none(self):
        """Test no anomalies detected."""
        data = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
        anomalies = TimeSeriesProcessor.detect_anomalies(data, threshold=3.0)
        assert np.sum(anomalies) == 0

    def test_detect_anomalies_constant(self):
        """Test anomaly detection on constant data."""
        data = np.array([5.0, 5.0, 5.0])
        anomalies = TimeSeriesProcessor.detect_anomalies(data)
        assert np.sum(anomalies) == 0

    def test_detect_anomalies_wrong_dimension(self):
        """Test anomaly detection with wrong dimensions."""
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Expected 1D array"):
            TimeSeriesProcessor.detect_anomalies(data)


class TestAPIDataIntegrator:
    """Test API data integration."""

    def test_process_api_response(self):
        """Test API response processing."""
        integrator = APIDataIntegrator()
        response = {'status': 200, 'data': {'value': 42}}
        vector = integrator.process_api_response(response)
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 100  # Default max_size

    def test_extract_numeric_data_simple(self):
        """Test extracting numeric data."""
        integrator = APIDataIntegrator()
        data = {'x': 10, 'y': 20}
        numeric = integrator.extract_numeric_data(data)
        assert 10.0 in numeric
        assert 20.0 in numeric

    def test_extract_numeric_data_nested(self):
        """Test extracting from nested structure."""
        integrator = APIDataIntegrator()
        data = {'outer': {'inner': 42}}
        numeric = integrator.extract_numeric_data(data)
        assert 42.0 in numeric

    def test_extract_numeric_data_with_list(self):
        """Test extracting from lists."""
        integrator = APIDataIntegrator()
        data = {'values': [1, 2, 3]}
        numeric = integrator.extract_numeric_data(data)
        assert 1.0 in numeric
        assert 2.0 in numeric
        assert 3.0 in numeric

    def test_extract_numeric_data_bool(self):
        """Test extracting boolean as numeric."""
        integrator = APIDataIntegrator()
        data = {'flag': True, 'switch': False}
        numeric = integrator.extract_numeric_data(data)
        assert 1.0 in numeric
        assert 0.0 in numeric

    def test_extract_numeric_data_empty(self):
        """Test extracting from empty data."""
        integrator = APIDataIntegrator()
        data = {}
        numeric = integrator.extract_numeric_data(data)
        assert len(numeric) == 1
        assert numeric[0] == 0.0

    def test_cache_data(self):
        """Test data caching."""
        integrator = APIDataIntegrator()
        integrator.cache_data('key1', {'value': 42})
        cached = integrator.get_cached_data('key1')
        assert cached['value'] == 42

    def test_get_cached_data_missing(self):
        """Test getting non-existent cached data."""
        integrator = APIDataIntegrator()
        cached = integrator.get_cached_data('nonexistent')
        assert cached is None

    def test_clear_cache(self):
        """Test cache clearing."""
        integrator = APIDataIntegrator()
        integrator.cache_data('key1', 'data1')
        integrator.cache_data('key2', 'data2')
        integrator.clear_cache()
        assert integrator.get_cached_data('key1') is None
        assert integrator.get_cached_data('key2') is None


class TestPreprocessDigitalInput:
    """Test main preprocessing function."""

    def test_preprocess_text(self):
        """Test preprocessing text input."""
        result = preprocess_digital_input("hello world", data_type='text', output_size=256)
        assert len(result) == 256
        assert isinstance(result, np.ndarray)

    def test_preprocess_json_string(self):
        """Test preprocessing JSON string."""
        json_str = '{"name": "test", "value": 42}'
        result = preprocess_digital_input(json_str, data_type='json', output_size=100)
        assert len(result) == 100

    def test_preprocess_json_dict(self):
        """Test preprocessing JSON dict."""
        data = {'name': 'test', 'value': 42}
        result = preprocess_digital_input(data, data_type='json', output_size=100)
        assert len(result) == 100

    def test_preprocess_timeseries_list(self):
        """Test preprocessing time-series from list."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = preprocess_digital_input(data, data_type='timeseries', output_size=10)
        assert len(result) == 10

    def test_preprocess_timeseries_array(self):
        """Test preprocessing time-series from array."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = preprocess_digital_input(data, data_type='timeseries', output_size=10)
        assert len(result) == 10

    def test_preprocess_api(self):
        """Test preprocessing API data."""
        data = {'status': 200, 'data': {'value': 42}}
        result = preprocess_digital_input(data, data_type='api', output_size=100)
        assert len(result) == 100

    def test_preprocess_invalid_type(self):
        """Test preprocessing with invalid type."""
        with pytest.raises(ValueError, match="Unknown data_type"):
            preprocess_digital_input("data", data_type='invalid')

    def test_preprocess_text_wrong_input_type(self):
        """Test text preprocessing with wrong input type."""
        with pytest.raises(ValueError, match="Expected string"):
            preprocess_digital_input(123, data_type='text')

    def test_preprocess_json_wrong_input_type(self):
        """Test JSON preprocessing with wrong input type."""
        with pytest.raises(ValueError, match="Expected dict"):
            preprocess_digital_input([1, 2, 3], data_type='json')

    def test_preprocess_timeseries_wrong_input_type(self):
        """Test time-series preprocessing with wrong input type."""
        with pytest.raises(ValueError, match="Expected array"):
            preprocess_digital_input("string", data_type='timeseries')

    def test_preprocess_api_wrong_input_type(self):
        """Test API preprocessing with wrong input type."""
        with pytest.raises(ValueError, match="Expected dict"):
            preprocess_digital_input("string", data_type='api')
