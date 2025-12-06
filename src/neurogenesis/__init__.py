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

from .neuron import NeuronBase, Dendrite, Axon, Soma, NeuronType, CompartmentType
from .glia import GliaCell, Astrocyte, Oligodendrocyte, Microglia, GliaType, GliaState
from .dna_bank import DNABank, GeneticParameters, ParameterCategory

__all__ = [
    'NeuronBase',
    'Dendrite',
    'Axon',
    'Soma',
    'NeuronType',
    'CompartmentType',
    'GliaCell',
    'Astrocyte',
    'Oligodendrocyte',
    'Microglia',
    'GliaType',
    'GliaState',
    'DNABank',
    'GeneticParameters',
    'ParameterCategory',
]

__version__ = '0.1.0'
