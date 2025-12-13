"""4D Slice Partitioning for distributing neurons across VPUs."""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class SlicePartitioner:
    """Partitions a 4D neural lattice into slices for parallel processing.
    
    This class implements various partitioning strategies to divide a 4D lattice
    into slices that can be assigned to Virtual Processing Units (VPUs). The key
    innovation is that each VPU processes an entire batch of neurons (e.g., a
    complete w-slice) rather than individual neurons.
    
    Example:
        For a 128x128x16x64 lattice partitioned by w-slice:
        - 64 VPUs (one per w-slice)
        - Each VPU processes 128x128x16 = 262,144 neurons as a batch
        - Per clock cycle: 262k neurons × 64 VPUs = 16.7M neurons
    """
    
    @staticmethod
    def partition_by_w_slice(
        lattice_shape: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int, int, int, int, int, int, int]]:
        """Partition the 4D lattice into w-slices.
        
        This is the primary partitioning strategy. Each VPU gets a complete
        3D slice at a fixed w coordinate, containing all neurons at that
        w position.
        
        Args:
            lattice_shape: 4-tuple defining lattice dimensions (x, y, z, w)
            
        Returns:
            List of slice bounds, each as an 8-tuple:
            (x_min, x_max, y_min, y_max, z_min, z_max, w_min, w_max)
        """
        x_size, y_size, z_size, w_size = lattice_shape
        partitions = []
        
        for w in range(w_size):
            slice_bounds = (
                0, x_size - 1,  # x range
                0, y_size - 1,  # y range
                0, z_size - 1,  # z range
                w, w,           # w fixed at this value
            )
            partitions.append(slice_bounds)
        
        logger.info(
            f"Created {len(partitions)} w-slices for lattice {lattice_shape}"
        )
        return partitions
    
    @staticmethod
    def partition_by_z_slice(
        lattice_shape: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int, int, int, int, int, int, int]]:
        """Partition the 4D lattice into z-slices.
        
        Alternative partitioning strategy where each VPU gets all neurons
        at a fixed z coordinate across all w values.
        
        Args:
            lattice_shape: 4-tuple defining lattice dimensions (x, y, z, w)
            
        Returns:
            List of slice bounds
        """
        x_size, y_size, z_size, w_size = lattice_shape
        partitions = []
        
        for z in range(z_size):
            slice_bounds = (
                0, x_size - 1,  # x range
                0, y_size - 1,  # y range
                z, z,           # z fixed at this value
                0, w_size - 1,  # w range
            )
            partitions.append(slice_bounds)
        
        logger.info(
            f"Created {len(partitions)} z-slices for lattice {lattice_shape}"
        )
        return partitions
    
    @staticmethod
    def partition_by_blocks(
        lattice_shape: Tuple[int, int, int, int],
        block_size: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int, int, int, int, int, int, int]]:
        """Partition the 4D lattice into blocks of specified size.
        
        This creates a grid of blocks, useful for more fine-grained
        load balancing when the lattice is not uniform in density.
        
        Args:
            lattice_shape: 4-tuple defining lattice dimensions (x, y, z, w)
            block_size: 4-tuple defining block dimensions
            
        Returns:
            List of slice bounds
        """
        x_size, y_size, z_size, w_size = lattice_shape
        bx, by, bz, bw = block_size
        partitions = []
        
        # Iterate through all blocks
        for w_start in range(0, w_size, bw):
            w_end = min(w_start + bw - 1, w_size - 1)
            for z_start in range(0, z_size, bz):
                z_end = min(z_start + bz - 1, z_size - 1)
                for y_start in range(0, y_size, by):
                    y_end = min(y_start + by - 1, y_size - 1)
                    for x_start in range(0, x_size, bx):
                        x_end = min(x_start + bx - 1, x_size - 1)
                        
                        slice_bounds = (
                            x_start, x_end,
                            y_start, y_end,
                            z_start, z_end,
                            w_start, w_end,
                        )
                        partitions.append(slice_bounds)
        
        logger.info(
            f"Created {len(partitions)} blocks of size {block_size} "
            f"for lattice {lattice_shape}"
        )
        return partitions
    
    @staticmethod
    def partition_adaptive(
        lattice_shape: Tuple[int, int, int, int],
        neuron_positions: List[Tuple[int, int, int, int]],
        target_vpus: int
    ) -> List[Tuple[int, int, int, int, int, int, int, int]]:
        """Adaptive partitioning based on actual neuron distribution.
        
        This creates partitions that balance the number of actual neurons
        per VPU, rather than just dividing space equally. Useful when
        neurons are not uniformly distributed.
        
        Args:
            lattice_shape: 4-tuple defining lattice dimensions (x, y, z, w)
            neuron_positions: List of (x, y, z, w) positions of actual neurons
            target_vpus: Target number of VPUs (partitions)
            
        Returns:
            List of slice bounds
        """
        # Simple implementation: sort neurons by w, then divide evenly
        if not neuron_positions:
            logger.warning("No neurons to partition")
            return []
        
        # Sort neurons by w coordinate
        sorted_neurons = sorted(neuron_positions, key=lambda pos: pos[3])
        neurons_per_vpu = len(sorted_neurons) // target_vpus
        
        partitions = []
        x_size, y_size, z_size, w_size = lattice_shape
        
        for i in range(target_vpus):
            start_idx = i * neurons_per_vpu
            end_idx = start_idx + neurons_per_vpu if i < target_vpus - 1 else len(sorted_neurons)
            
            if start_idx >= len(sorted_neurons):
                break
            
            # Find bounds of neurons in this partition
            neurons_in_partition = sorted_neurons[start_idx:end_idx]
            
            x_coords = [n[0] for n in neurons_in_partition]
            y_coords = [n[1] for n in neurons_in_partition]
            z_coords = [n[2] for n in neurons_in_partition]
            w_coords = [n[3] for n in neurons_in_partition]
            
            slice_bounds = (
                min(x_coords), max(x_coords),
                min(y_coords), max(y_coords),
                min(z_coords), max(z_coords),
                min(w_coords), max(w_coords),
            )
            partitions.append(slice_bounds)
        
        logger.info(
            f"Created {len(partitions)} adaptive partitions "
            f"for {len(neuron_positions)} neurons"
        )
        return partitions
    
    @staticmethod
    def get_partition_info(
        slice_bounds: Tuple[int, int, int, int, int, int, int, int]
    ) -> dict:
        """Get information about a partition slice.
        
        Args:
            slice_bounds: 8-tuple defining the slice bounds
            
        Returns:
            Dictionary with partition information
        """
        x_min, x_max, y_min, y_max, z_min, z_max, w_min, w_max = slice_bounds
        
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1
        z_size = z_max - z_min + 1
        w_size = w_max - w_min + 1
        
        return {
            "bounds": {
                "x": (x_min, x_max),
                "y": (y_min, y_max),
                "z": (z_min, z_max),
                "w": (w_min, w_max),
            },
            "size": {
                "x": x_size,
                "y": y_size,
                "z": z_size,
                "w": w_size,
            },
            "volume": x_size * y_size * z_size * w_size,
        }
    
    @staticmethod
    def visualize_partitions(
        partitions: List[Tuple[int, int, int, int, int, int, int, int]],
        lattice_shape: Tuple[int, int, int, int]
    ) -> str:
        """Create a text visualization of the partitioning scheme.
        
        Args:
            partitions: List of slice bounds
            lattice_shape: Overall lattice dimensions
            
        Returns:
            String with ASCII visualization
        """
        lines = []
        lines.append(f"Lattice Shape: {lattice_shape}")
        lines.append(f"Number of Partitions: {len(partitions)}")
        lines.append("")
        
        total_volume = 0
        for i, partition in enumerate(partitions):
            info = SlicePartitioner.get_partition_info(partition)
            lines.append(f"Partition {i}:")
            lines.append(f"  Bounds: x={info['bounds']['x']}, y={info['bounds']['y']}, "
                        f"z={info['bounds']['z']}, w={info['bounds']['w']}")
            lines.append(f"  Volume: {info['volume']} positions")
            total_volume += info['volume']
        
        lines.append("")
        lattice_volume = lattice_shape[0] * lattice_shape[1] * lattice_shape[2] * lattice_shape[3]
        coverage = (total_volume / lattice_volume * 100) if lattice_volume > 0 else 0
        lines.append(f"Total Coverage: {total_volume}/{lattice_volume} ({coverage:.1f}%)")
        
        return "\n".join(lines)
