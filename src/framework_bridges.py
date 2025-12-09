"""Bridge modules for integration with popular deep learning frameworks.

This module provides interfaces to convert between 4D Neural Cognition
models and PyTorch/TensorFlow/Keras models, enabling hybrid approaches.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import json

# Try importing optional dependencies
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class BrainToPyTorchConverter:
    """Convert 4D brain models to PyTorch modules."""
    
    @staticmethod
    def to_pytorch_module(brain_model: Any) -> Optional['nn.Module']:
        """Convert brain model to PyTorch module.
        
        Args:
            brain_model: BrainModel instance
            
        Returns:
            PyTorch module or None if PyTorch not available
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        return PyTorchBrainWrapper(brain_model)
    
    @staticmethod
    def export_weights(brain_model: Any) -> Dict[str, np.ndarray]:
        """Export brain model weights as numpy arrays.
        
        Args:
            brain_model: BrainModel instance
            
        Returns:
            Dictionary of weight arrays
        """
        weights = {}
        
        # Export synaptic weights
        if hasattr(brain_model, 'synapses'):
            synapse_weights = []
            pre_ids = []
            post_ids = []
            
            for syn in brain_model.synapses:
                synapse_weights.append(syn.weight)
                pre_ids.append(syn.pre_id)
                post_ids.append(syn.post_id)
            
            weights['synapse_weights'] = np.array(synapse_weights)
            weights['pre_neuron_ids'] = np.array(pre_ids)
            weights['post_neuron_ids'] = np.array(post_ids)
        
        # Export neuron parameters
        if hasattr(brain_model, 'neurons'):
            v_membranes = []
            neuron_ids = []
            
            for neuron in brain_model.neurons.values():
                v_membranes.append(neuron.v_membrane)
                neuron_ids.append(neuron.id)
            
            weights['v_membranes'] = np.array(v_membranes)
            weights['neuron_ids'] = np.array(neuron_ids)
        
        return weights


class PyTorchBrainWrapper(nn.Module if PYTORCH_AVAILABLE else object):
    """PyTorch module wrapper for brain model."""
    
    def __init__(self, brain_model: Any):
        """Initialize wrapper.
        
        Args:
            brain_model: BrainModel instance to wrap
        """
        if PYTORCH_AVAILABLE:
            super().__init__()
        
        self.brain_model = brain_model
        
        if PYTORCH_AVAILABLE:
            self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize PyTorch parameters from brain model."""
        if not hasattr(self.brain_model, 'synapses'):
            return
        
        # Create weight matrix
        num_neurons = len(self.brain_model.neurons)
        
        # Initialize sparse weight matrix
        weights = np.zeros((num_neurons, num_neurons))
        
        for syn in self.brain_model.synapses:
            if syn.pre_id < num_neurons and syn.post_id < num_neurons:
                weights[syn.post_id, syn.pre_id] = syn.weight
        
        # Register as parameter
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass through brain model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Simple linear transformation using synaptic weights
        return torch.matmul(x, self.weights)


class TensorFlowBrainWrapper:
    """TensorFlow/Keras layer wrapper for brain model."""
    
    def __init__(self, brain_model: Any):
        """Initialize wrapper.
        
        Args:
            brain_model: BrainModel instance to wrap
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        self.brain_model = brain_model
        self.layer = self._create_keras_layer()
    
    def _create_keras_layer(self) -> 'tf.keras.layers.Layer':
        """Create Keras layer from brain model."""
        class BrainLayer(tf.keras.layers.Layer):
            def __init__(self, brain_model, **kwargs):
                super().__init__(**kwargs)
                self.brain_model = brain_model
                self.num_neurons = len(brain_model.neurons)
            
            def build(self, input_shape):
                # Initialize weights from brain model
                weights = np.zeros((input_shape[-1], self.num_neurons))
                
                for syn in self.brain_model.synapses:
                    if (syn.pre_id < input_shape[-1] and 
                        syn.post_id < self.num_neurons):
                        weights[syn.pre_id, syn.post_id] = syn.weight
                
                self.kernel = self.add_weight(
                    name='kernel',
                    shape=(input_shape[-1], self.num_neurons),
                    initializer=tf.constant_initializer(weights),
                    trainable=True
                )
                super().build(input_shape)
            
            def call(self, inputs):
                return tf.matmul(inputs, self.kernel)
        
        return BrainLayer(self.brain_model)
    
    def get_layer(self) -> 'tf.keras.layers.Layer':
        """Get the Keras layer."""
        return self.layer


class ModelExporter:
    """Export brain models to standard formats."""
    
    @staticmethod
    def export_to_onnx(
        brain_model: Any,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 10)
    ) -> None:
        """Export model to ONNX format.
        
        Args:
            brain_model: BrainModel instance
            output_path: Path to save ONNX model
            input_shape: Input shape for model
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for ONNX export")
        
        # Convert to PyTorch
        pytorch_model = PyTorchBrainWrapper(brain_model)
        pytorch_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
    
    @staticmethod
    def export_to_saved_model(
        brain_model: Any,
        output_path: str
    ) -> None:
        """Export model to TensorFlow SavedModel format.
        
        Args:
            brain_model: BrainModel instance
            output_path: Path to save model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for SavedModel export")
        
        # Convert to Keras
        wrapper = TensorFlowBrainWrapper(brain_model)
        layer = wrapper.get_layer()
        
        # Create model
        input_shape = (len(brain_model.neurons),)
        inputs = tf.keras.Input(shape=input_shape)
        outputs = layer(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Save
        model.save(output_path)
    
    @staticmethod
    def export_to_json(
        brain_model: Any,
        output_path: str
    ) -> None:
        """Export model architecture to JSON.
        
        Args:
            brain_model: BrainModel instance
            output_path: Path to save JSON
        """
        data = {
            'neurons': [],
            'synapses': [],
            'config': brain_model.config if hasattr(brain_model, 'config') else {}
        }
        
        # Export neurons
        if hasattr(brain_model, 'neurons'):
            for neuron_id, neuron in brain_model.neurons.items():
                data['neurons'].append({
                    'id': neuron.id,
                    'x': neuron.x,
                    'y': neuron.y,
                    'z': neuron.z,
                    'w': neuron.w,
                    'type': neuron.neuron_type,
                    'model': neuron.model_type
                })
        
        # Export synapses
        if hasattr(brain_model, 'synapses'):
            for syn in brain_model.synapses:
                data['synapses'].append({
                    'pre_id': syn.pre_id,
                    'post_id': syn.post_id,
                    'weight': syn.weight,
                    'delay': syn.delay
                })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


class ModelImporter:
    """Import models from other frameworks."""
    
    @staticmethod
    def from_pytorch(
        pytorch_model: Any,
        brain_model: Any
    ) -> None:
        """Import weights from PyTorch model.
        
        Args:
            pytorch_model: PyTorch model
            brain_model: BrainModel to import into
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Extract weights from first linear layer
        for name, param in pytorch_model.named_parameters():
            if 'weight' in name.lower():
                weights = param.detach().cpu().numpy()
                
                # Update synaptic weights
                if hasattr(brain_model, 'synapses'):
                    for syn in brain_model.synapses:
                        if (syn.post_id < weights.shape[0] and 
                            syn.pre_id < weights.shape[1]):
                            syn.weight = float(weights[syn.post_id, syn.pre_id])
                
                break
    
    @staticmethod
    def from_tensorflow(
        tf_model: Any,
        brain_model: Any
    ) -> None:
        """Import weights from TensorFlow/Keras model.
        
        Args:
            tf_model: TensorFlow/Keras model
            brain_model: BrainModel to import into
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Get weights from first layer
        if hasattr(tf_model, 'layers') and len(tf_model.layers) > 0:
            weights = tf_model.layers[0].get_weights()
            
            if len(weights) > 0:
                weight_matrix = weights[0]
                
                # Update synaptic weights
                if hasattr(brain_model, 'synapses'):
                    for syn in brain_model.synapses:
                        if (syn.pre_id < weight_matrix.shape[0] and 
                            syn.post_id < weight_matrix.shape[1]):
                            syn.weight = float(weight_matrix[syn.pre_id, syn.post_id])


class HybridModel:
    """Hybrid model combining brain model with deep learning."""
    
    def __init__(
        self,
        brain_model: Any,
        framework: str = "pytorch"
    ):
        """Initialize hybrid model.
        
        Args:
            brain_model: BrainModel instance
            framework: Framework to use ('pytorch' or 'tensorflow')
        """
        self.brain_model = brain_model
        self.framework = framework
        
        if framework == "pytorch" and PYTORCH_AVAILABLE:
            self.dl_model = self._create_pytorch_model()
        elif framework == "tensorflow" and TENSORFLOW_AVAILABLE:
            self.dl_model = self._create_tensorflow_model()
        else:
            raise ValueError(f"Framework {framework} not available or not supported")
    
    def _create_pytorch_model(self) -> Optional[Any]:
        """Create PyTorch model component."""
        class HybridPyTorch(nn.Module):
            def __init__(self, brain_wrapper):
                super().__init__()
                self.brain_layer = brain_wrapper
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 10)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                # Process through brain model
                x = self.brain_layer(x)
                # Additional DL layers
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        brain_wrapper = PyTorchBrainWrapper(self.brain_model)
        return HybridPyTorch(brain_wrapper)
    
    def _create_tensorflow_model(self) -> Optional[Any]:
        """Create TensorFlow model component."""
        brain_wrapper = TensorFlowBrainWrapper(self.brain_model)
        brain_layer = brain_wrapper.get_layer()
        
        # Build hybrid model
        inputs = tf.keras.Input(shape=(len(self.brain_model.neurons),))
        x = brain_layer(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def get_model(self) -> Any:
        """Get the hybrid deep learning model."""
        return self.dl_model


def get_available_frameworks() -> Dict[str, bool]:
    """Get information about available frameworks.
    
    Returns:
        Dictionary mapping framework names to availability
    """
    return {
        'pytorch': PYTORCH_AVAILABLE,
        'tensorflow': TENSORFLOW_AVAILABLE
    }
