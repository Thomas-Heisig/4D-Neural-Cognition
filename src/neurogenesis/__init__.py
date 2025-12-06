"""
Neurogenesis Module - Self-organizing Virtual Brain System

This module implements a flexible development schema for a self-organizing
virtual brain system based on biological principles. It provides components
for neuron development, glia cell integration, and DNA-based parameter management.

Hauptmodule / Main Modules:
- neuron: Base neuron structures and components
- glia: Glia cell types (astrocytes, oligodendrocytes, etc.)
- dna_bank: Central DNA/parameter bank for cell configuration
"""

from .neuron import NeuronBase, Dendrite, Axon, Soma
from .glia import GliaCell, Astrocyte, Oligodendrocyte, Microglia
from .dna_bank import DNABank, GeneticParameters

__all__ = [
    'NeuronBase',
    'Dendrite',
    'Axon',
    'Soma',
    'GliaCell',
    'Astrocyte',
    'Oligodendrocyte',
    'Microglia',
    'DNABank',
    'GeneticParameters',
]

__version__ = '0.1.0'
