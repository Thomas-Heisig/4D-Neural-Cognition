"""Digital Sense 2.0 - Direct Neural API for external data integration."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DirectNeuralAPI:
    """Direct Neural API that connects external data sources to neurons.
    
    This is the enhanced "Digital Sense 2.0" that provides a universal
    interface for connecting any external data source directly to the
    neural network. It supports various data streams including:
    - WebSocket live data (stock prices, IoT sensors)
    - Database query results (SQL, GraphQL)
    - File system monitoring
    - REST/gRPC API responses
    
    The key innovation is that neural activity patterns can trigger
    API calls, and API responses can be encoded back into neural inputs,
    creating a bidirectional interface between the neural network and
    external systems.
    
    Attributes:
        data_streams: Active data streams
        api_endpoints: Registered API endpoints
        encoders: Functions to encode data to neural patterns
        decoders: Functions to decode neural patterns to data
    """
    
    def __init__(self):
        """Initialize the Direct Neural API."""
        self.data_streams: Dict[str, DataStream] = {}
        self.api_endpoints: List[APIEndpoint] = []
        
        # Encoding/Decoding functions
        self.encoders: Dict[str, Callable] = {}
        self.decoders: Dict[str, Callable] = {}
        
        # Statistics
        self.statistics = {
            "streams_connected": 0,
            "api_calls_made": 0,
            "data_encoded": 0,
            "data_decoded": 0,
        }
        
        # Default encoders/decoders
        self._register_default_codecs()
        
        logger.info("Initialized DirectNeuralAPI (Digital Sense 2.0)")
    
    def _register_default_codecs(self) -> None:
        """Register default encoding and decoding functions."""
        # Simple value to spike rate encoder
        self.register_encoder(
            "value_to_rate",
            lambda value: float(value) * 100.0  # Convert to Hz
        )
        
        # Array to neural pattern
        self.register_encoder(
            "array_to_pattern",
            lambda array: np.array(array, dtype=float)
        )
        
        # Neural pattern to value
        self.register_decoder(
            "rate_to_value",
            lambda rate: float(rate) / 100.0
        )
        
        # Neural pattern to array
        self.register_decoder(
            "pattern_to_array",
            lambda pattern: pattern.tolist() if isinstance(pattern, np.ndarray) else list(pattern)
        )
    
    def register_encoder(self, name: str, encoder_fn: Callable) -> None:
        """Register an encoder function.
        
        Args:
            name: Name for this encoder
            encoder_fn: Function that takes data and returns neural input
        """
        self.encoders[name] = encoder_fn
        logger.info(f"Registered encoder: {name}")
    
    def register_decoder(self, name: str, decoder_fn: Callable) -> None:
        """Register a decoder function.
        
        Args:
            name: Name for this decoder
            decoder_fn: Function that takes neural pattern and returns data
        """
        self.decoders[name] = decoder_fn
        logger.info(f"Registered decoder: {name}")
    
    def connect_data_stream(
        self,
        stream_id: str,
        stream_type: str,
        config: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> DataStream:
        """Connect an external data stream.
        
        Args:
            stream_id: Unique identifier for this stream
            stream_type: Type of stream (websocket, database, file, api)
            config: Configuration dictionary for the stream
            callback: Optional callback when data arrives
            
        Returns:
            DataStream object
        """
        stream = DataStream(stream_id, stream_type, config, callback)
        self.data_streams[stream_id] = stream
        self.statistics["streams_connected"] += 1
        
        logger.info(f"Connected data stream: {stream_id} (type: {stream_type})")
        return stream
    
    def disconnect_data_stream(self, stream_id: str) -> bool:
        """Disconnect a data stream.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            True if stream was found and disconnected
        """
        if stream_id in self.data_streams:
            stream = self.data_streams[stream_id]
            stream.disconnect()
            del self.data_streams[stream_id]
            logger.info(f"Disconnected data stream: {stream_id}")
            return True
        return False
    
    def neural_api_call(
        self,
        endpoint: str,
        neural_pattern: np.ndarray,
        decoder: str = "pattern_to_array"
    ) -> Any:
        """Make an API call triggered by neural activity.
        
        This is the key method that allows neural activity to trigger
        external actions. The neural pattern is decoded and used to
        construct an API call.
        
        Args:
            endpoint: API endpoint to call
            neural_pattern: Neural activity pattern that triggered the call
            decoder: Name of decoder to use
            
        Returns:
            API response data
        """
        self.statistics["api_calls_made"] += 1
        
        # Decode neural pattern
        if decoder not in self.decoders:
            logger.warning(f"Unknown decoder: {decoder}, using default")
            decoder = "pattern_to_array"
        
        decoded_data = self.decoders[decoder](neural_pattern)
        self.statistics["data_decoded"] += 1
        
        # Find endpoint
        api_endpoint = None
        for ep in self.api_endpoints:
            if ep.endpoint == endpoint:
                api_endpoint = ep
                break
        
        if api_endpoint is None:
            logger.warning(f"Unknown endpoint: {endpoint}")
            return None
        
        # Execute API call
        try:
            result = api_endpoint.execute(decoded_data)
            logger.debug(f"API call to {endpoint} succeeded")
            return result
        except Exception as e:
            logger.error(f"API call to {endpoint} failed: {e}")
            return None
    
    def encode_to_neural_input(
        self,
        data: Any,
        encoder: str = "value_to_rate"
    ) -> np.ndarray:
        """Encode external data into neural input format.
        
        Args:
            data: External data to encode
            encoder: Name of encoder to use
            
        Returns:
            Neural input pattern as NumPy array
        """
        self.statistics["data_encoded"] += 1
        
        if encoder not in self.encoders:
            logger.warning(f"Unknown encoder: {encoder}, using default")
            encoder = "value_to_rate"
        
        try:
            encoded = self.encoders[encoder](data)
            if not isinstance(encoded, np.ndarray):
                encoded = np.array([encoded])
            return encoded
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return np.array([0.0])
    
    def decode_neural_to_sql(self, neural_pattern: np.ndarray) -> str:
        """Decode neural pattern into SQL query.
        
        This is a placeholder for future implementation where neural
        patterns could be interpreted as database queries.
        
        Args:
            neural_pattern: Neural activity pattern
            
        Returns:
            SQL query string
        """
        # TODO: Implement neural-to-SQL decoding
        # For now, return a simple template
        return "SELECT * FROM data LIMIT 10"
    
    def execute_query(self, query: str) -> List[Dict]:
        """Execute a database query.
        
        Placeholder for database integration.
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as list of dictionaries
        """
        # TODO: Implement actual database connection
        logger.debug(f"Query execution (mock): {query}")
        return [{"id": 1, "value": 42.0}]
    
    def register_api_endpoint(
        self,
        endpoint: str,
        method: str,
        handler: Callable
    ) -> None:
        """Register an API endpoint.
        
        Args:
            endpoint: Endpoint path
            method: HTTP method (GET, POST, etc.)
            handler: Function to handle requests
        """
        api_endpoint = APIEndpoint(endpoint, method, handler)
        self.api_endpoints.append(api_endpoint)
        logger.info(f"Registered API endpoint: {method} {endpoint}")
    
    def get_statistics(self) -> dict:
        """Get API statistics.
        
        Returns:
            Dictionary with usage metrics
        """
        stats = self.statistics.copy()
        stats["active_streams"] = len(self.data_streams)
        stats["registered_endpoints"] = len(self.api_endpoints)
        stats["registered_encoders"] = len(self.encoders)
        stats["registered_decoders"] = len(self.decoders)
        
        return stats


class DataStream:
    """Represents an external data stream."""
    
    def __init__(
        self,
        stream_id: str,
        stream_type: str,
        config: Dict[str, Any],
        callback: Optional[Callable] = None
    ):
        """Initialize a data stream.
        
        Args:
            stream_id: Unique identifier
            stream_type: Type of stream
            config: Configuration dictionary
            callback: Optional data callback
        """
        self.stream_id = stream_id
        self.stream_type = stream_type
        self.config = config
        self.callback = callback
        self.is_active = True
        
        # Initialize based on type
        if stream_type == "websocket":
            self._init_websocket()
        elif stream_type == "database":
            self._init_database()
        elif stream_type == "file":
            self._init_file_monitor()
        elif stream_type == "api":
            self._init_api_poller()
        else:
            logger.warning(f"Unknown stream type: {stream_type}")
    
    def _init_websocket(self) -> None:
        """Initialize WebSocket connection."""
        # TODO: Implement WebSocket connection
        logger.info(f"WebSocket stream initialized (mock): {self.stream_id}")
    
    def _init_database(self) -> None:
        """Initialize database connection."""
        # TODO: Implement database connection
        logger.info(f"Database stream initialized (mock): {self.stream_id}")
    
    def _init_file_monitor(self) -> None:
        """Initialize file system monitor."""
        # TODO: Implement file monitoring
        logger.info(f"File monitor initialized (mock): {self.stream_id}")
    
    def _init_api_poller(self) -> None:
        """Initialize API polling."""
        # TODO: Implement API polling
        logger.info(f"API poller initialized (mock): {self.stream_id}")
    
    def disconnect(self) -> None:
        """Disconnect the stream."""
        self.is_active = False
        logger.info(f"Stream disconnected: {self.stream_id}")
    
    def send_data(self, data: Any) -> None:
        """Send data through this stream.
        
        Args:
            data: Data to send
        """
        if self.callback is not None:
            try:
                self.callback(data)
            except Exception as e:
                logger.error(f"Stream callback error: {e}")


class APIEndpoint:
    """Represents an API endpoint."""
    
    def __init__(self, endpoint: str, method: str, handler: Callable):
        """Initialize an API endpoint.
        
        Args:
            endpoint: Endpoint path
            method: HTTP method
            handler: Request handler function
        """
        self.endpoint = endpoint
        self.method = method
        self.handler = handler
    
    def execute(self, data: Any) -> Any:
        """Execute the endpoint handler.
        
        Args:
            data: Request data
            
        Returns:
            Response data
        """
        return self.handler(data)
