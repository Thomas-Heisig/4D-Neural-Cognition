"""Hardware abstraction layer for Virtual Neuromorphic Clock (VNC) system."""

from .virtual_clock import GlobalVirtualClock
from .virtual_processing_unit import VirtualProcessingUnit
from .slice_partitioner import SlicePartitioner
from .virtual_io_expander import VirtualIOExpander

__all__ = [
    "GlobalVirtualClock",
    "VirtualProcessingUnit",
    "SlicePartitioner",
    "VirtualIOExpander",
]
