"""Virtual I/O Expander for managing large numbers of virtual I/O ports."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VirtualIOExpander:
    """Virtual I/O Expander that provides massive I/O port expansion.
    
    This class implements virtual port expansion that allows the system to
    have far more I/O ports than physically possible. It uses time-multiplexing
    and dynamic mapping to create the illusion of hundreds of thousands of
    simultaneous I/O ports.
    
    Example:
        With base_io_width=1024 and expansion_factor=256:
        - 262,144 virtual ports available
        - Dynamic mapping to physical pins
        - Batch read/write operations
    
    Attributes:
        base_io_width: Number of physical I/O pins
        expansion_factor: Multiplier for virtual expansion
        virtual_ports: Total number of virtual ports
        port_mapping: Mapping from virtual to physical ports
    """
    
    def __init__(self, base_io_width: int = 1024, expansion_factor: int = 256):
        """Initialize the Virtual I/O Expander.
        
        Args:
            base_io_width: Number of physical I/O pins (default: 1024)
            expansion_factor: Virtual expansion factor (default: 256)
        """
        self.base_io_width = base_io_width
        self.expansion_factor = expansion_factor
        self.virtual_ports = base_io_width * expansion_factor
        
        # Port mapping: virtual_port_id -> physical_pin_id
        self.port_mapping: Dict[int, int] = {}
        
        # Port data buffers
        self._input_buffer: Dict[int, float] = {}
        self._output_buffer: Dict[int, float] = {}
        
        # Port listeners for callbacks
        self._port_listeners: Dict[int, List[Callable]] = {}
        
        # Statistics
        self.statistics = {
            "reads": 0,
            "writes": 0,
            "mappings_created": 0,
        }
        
        logger.info(
            f"Initialized VirtualIOExpander: "
            f"{self.virtual_ports} virtual ports "
            f"({base_io_width} × {expansion_factor})"
        )
    
    def map_virtual_to_physical(
        self, 
        virtual_port: int, 
        physical_pin: int
    ) -> None:
        """Map a virtual port to a physical pin.
        
        Args:
            virtual_port: Virtual port ID (0 to virtual_ports-1)
            physical_pin: Physical pin ID (0 to base_io_width-1)
            
        Raises:
            ValueError: If port or pin IDs are out of range
        """
        if not 0 <= virtual_port < self.virtual_ports:
            raise ValueError(
                f"Virtual port {virtual_port} out of range "
                f"(0-{self.virtual_ports-1})"
            )
        
        if not 0 <= physical_pin < self.base_io_width:
            raise ValueError(
                f"Physical pin {physical_pin} out of range "
                f"(0-{self.base_io_width-1})"
            )
        
        self.port_mapping[virtual_port] = physical_pin
        self.statistics["mappings_created"] += 1
        logger.debug(f"Mapped virtual port {virtual_port} -> physical pin {physical_pin}")
    
    def auto_map_ports(self, num_ports: int) -> List[int]:
        """Automatically map a range of virtual ports to physical pins.
        
        Uses a round-robin strategy to distribute virtual ports across
        physical pins.
        
        Args:
            num_ports: Number of virtual ports to map
            
        Returns:
            List of virtual port IDs that were mapped
        """
        mapped_ports = []
        
        for i in range(num_ports):
            virtual_port = i % self.virtual_ports
            physical_pin = i % self.base_io_width
            
            if virtual_port not in self.port_mapping:
                self.map_virtual_to_physical(virtual_port, physical_pin)
            
            mapped_ports.append(virtual_port)
        
        logger.info(f"Auto-mapped {len(mapped_ports)} virtual ports")
        return mapped_ports
    
    def write_virtual_port(self, virtual_port: int, value: float) -> None:
        """Write a value to a virtual port.
        
        Args:
            virtual_port: Virtual port ID
            value: Value to write
            
        Raises:
            ValueError: If port is not mapped
        """
        if virtual_port not in self.port_mapping:
            raise ValueError(f"Virtual port {virtual_port} not mapped")
        
        self._output_buffer[virtual_port] = value
        self.statistics["writes"] += 1
        
        # Trigger listeners
        if virtual_port in self._port_listeners:
            for listener in self._port_listeners[virtual_port]:
                try:
                    listener(virtual_port, value)
                except Exception as e:
                    logger.error(f"Port listener error: {e}")
    
    def read_virtual_port(self, virtual_port: int) -> float:
        """Read a value from a virtual port.
        
        Args:
            virtual_port: Virtual port ID
            
        Returns:
            Current value at the port
            
        Raises:
            ValueError: If port is not mapped
        """
        if virtual_port not in self.port_mapping:
            raise ValueError(f"Virtual port {virtual_port} not mapped")
        
        self.statistics["reads"] += 1
        return self._input_buffer.get(virtual_port, 0.0)
    
    def read_virtual_bus(self, virtual_ports: List[int]) -> np.ndarray:
        """Read from multiple virtual ports as a batch.
        
        This is the key operation that enables efficient parallel I/O.
        Reads are aggregated and time-multiplexed across physical pins.
        
        Args:
            virtual_ports: List of virtual port IDs to read
            
        Returns:
            NumPy array of values from the ports
        """
        values = []
        
        for port in virtual_ports:
            try:
                value = self.read_virtual_port(port)
                values.append(value)
            except ValueError:
                # Port not mapped, return 0
                values.append(0.0)
        
        return np.array(values)
    
    def write_virtual_bus(
        self, 
        virtual_ports: List[int], 
        values: np.ndarray
    ) -> None:
        """Write to multiple virtual ports as a batch.
        
        Args:
            virtual_ports: List of virtual port IDs to write
            values: NumPy array of values to write
            
        Raises:
            ValueError: If lengths don't match
        """
        if len(virtual_ports) != len(values):
            raise ValueError(
                f"Port count ({len(virtual_ports)}) must match "
                f"value count ({len(values)})"
            )
        
        for port, value in zip(virtual_ports, values):
            try:
                self.write_virtual_port(port, float(value))
            except ValueError:
                # Auto-map unmapped ports
                physical_pin = port % self.base_io_width
                self.map_virtual_to_physical(port, physical_pin)
                self.write_virtual_port(port, float(value))
    
    def set_input_values(self, port_values: Dict[int, float]) -> None:
        """Set input values for multiple ports (simulating external inputs).
        
        Args:
            port_values: Dictionary mapping virtual port IDs to values
        """
        for port, value in port_values.items():
            if port < self.virtual_ports:
                self._input_buffer[port] = value
    
    def get_output_values(self) -> Dict[int, float]:
        """Get all current output values.
        
        Returns:
            Dictionary mapping virtual port IDs to output values
        """
        return self._output_buffer.copy()
    
    def add_port_listener(
        self, 
        virtual_port: int, 
        callback: Callable[[int, float], None]
    ) -> None:
        """Add a listener callback for a virtual port.
        
        The callback will be called whenever the port is written to.
        
        Args:
            virtual_port: Virtual port ID to listen to
            callback: Callback function with signature (port_id, value)
        """
        if virtual_port not in self._port_listeners:
            self._port_listeners[virtual_port] = []
        
        self._port_listeners[virtual_port].append(callback)
        logger.debug(f"Added listener to virtual port {virtual_port}")
    
    def remove_port_listener(
        self, 
        virtual_port: int, 
        callback: Callable[[int, float], None]
    ) -> bool:
        """Remove a listener callback from a virtual port.
        
        Args:
            virtual_port: Virtual port ID
            callback: Callback function to remove
            
        Returns:
            True if listener was found and removed
        """
        if virtual_port in self._port_listeners:
            try:
                self._port_listeners[virtual_port].remove(callback)
                logger.debug(f"Removed listener from virtual port {virtual_port}")
                return True
            except ValueError:
                pass
        return False
    
    def clear_buffers(self) -> None:
        """Clear all input and output buffers."""
        self._input_buffer.clear()
        self._output_buffer.clear()
        logger.debug("Cleared I/O buffers")
    
    def get_statistics(self) -> dict:
        """Get I/O statistics.
        
        Returns:
            Dictionary with I/O performance metrics
        """
        stats = self.statistics.copy()
        stats["virtual_ports"] = self.virtual_ports
        stats["base_io_width"] = self.base_io_width
        stats["expansion_factor"] = self.expansion_factor
        stats["ports_mapped"] = len(self.port_mapping)
        stats["active_listeners"] = sum(len(listeners) for listeners in self._port_listeners.values())
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset I/O statistics."""
        self.statistics = {
            "reads": 0,
            "writes": 0,
            "mappings_created": len(self.port_mapping),  # Keep existing mappings count
        }
