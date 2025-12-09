"""Data export tools for various formats (NWB, NumPy, CSV, MATLAB).

This module provides tools for exporting simulation data to:
- Neurodata Without Borders (NWB) format
- NumPy arrays and pickles
- CSV files for analysis
- MATLAB .mat files
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np

try:
    from .brain_model import BrainModel
    from .simulation import Simulation
except ImportError:
    from brain_model import BrainModel
    from simulation import Simulation


class NumpyExporter:
    """Export simulation data to NumPy format."""
    
    @staticmethod
    def export_network_structure(model: BrainModel, output_path: Union[str, Path]) -> None:
        """Export network structure as NumPy arrays.
        
        Args:
            model: Brain model to export
            output_path: Output file path (.npz)
        """
        output_path = Path(output_path)
        
        # Collect neuron data
        neuron_ids = sorted(model.neurons.keys())
        n_neurons = len(neuron_ids)
        
        positions = np.array([model.neurons[nid].position for nid in neuron_ids])
        v_membrane = np.array([model.neurons[nid].v_membrane for nid in neuron_ids])
        health = np.array([model.neurons[nid].health for nid in neuron_ids])
        age = np.array([model.neurons[nid].age for nid in neuron_ids])
        areas = np.array([model.neurons[nid].area for nid in neuron_ids], dtype=object)
        
        # Collect synapse data
        n_synapses = len(model.synapses)
        pre_ids = np.array([s.pre_id for s in model.synapses])
        post_ids = np.array([s.post_id for s in model.synapses])
        weights = np.array([s.weight for s in model.synapses])
        delays = np.array([s.delay for s in model.synapses])
        
        # Save to compressed .npz file
        np.savez_compressed(
            output_path,
            neuron_ids=neuron_ids,
            positions=positions,
            v_membrane=v_membrane,
            health=health,
            age=age,
            areas=areas,
            pre_ids=pre_ids,
            post_ids=post_ids,
            weights=weights,
            delays=delays,
            n_neurons=n_neurons,
            n_synapses=n_synapses,
            lattice_size=model.lattice_size
        )
        
        print(f"Network structure exported to {output_path}")
        print(f"  Neurons: {n_neurons}")
        print(f"  Synapses: {n_synapses}")
    
    @staticmethod
    def export_spike_trains(
        spike_history: Dict[int, List[int]], 
        output_path: Union[str, Path],
        dt: float = 1.0
    ) -> None:
        """Export spike trains as NumPy arrays.
        
        Args:
            spike_history: Dictionary mapping neuron_id to list of spike times
            output_path: Output file path (.npz)
            dt: Time step size in ms
        """
        output_path = Path(output_path)
        
        # Find all neurons and time range
        neuron_ids = sorted(spike_history.keys())
        all_times = [t for times in spike_history.values() for t in times]
        max_time = max(all_times) if all_times else 0
        
        # Create binary spike matrix (neurons x time)
        n_neurons = len(neuron_ids)
        n_timesteps = max_time + 1
        spike_matrix = np.zeros((n_neurons, n_timesteps), dtype=bool)
        
        for i, nid in enumerate(neuron_ids):
            for spike_time in spike_history[nid]:
                if 0 <= spike_time < n_timesteps:
                    spike_matrix[i, spike_time] = True
        
        # Calculate firing rates
        firing_rates = np.sum(spike_matrix, axis=1) / (n_timesteps * dt / 1000.0)  # Convert to Hz
        
        np.savez_compressed(
            output_path,
            spike_matrix=spike_matrix,
            neuron_ids=neuron_ids,
            firing_rates=firing_rates,
            dt=dt,
            duration_ms=n_timesteps * dt
        )
        
        print(f"Spike trains exported to {output_path}")
        print(f"  Neurons: {n_neurons}")
        print(f"  Duration: {n_timesteps * dt:.1f} ms")
        print(f"  Total spikes: {np.sum(spike_matrix)}")


class CSVExporter:
    """Export simulation data to CSV format."""
    
    @staticmethod
    def export_neurons(model: BrainModel, output_path: Union[str, Path]) -> None:
        """Export neuron properties to CSV.
        
        Args:
            model: Brain model to export
            output_path: Output file path (.csv)
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'neuron_id', 'x', 'y', 'z', 'w', 'area',
                'v_membrane', 'health', 'age', 'last_spike_time'
            ])
            
            # Write neuron data
            for nid in sorted(model.neurons.keys()):
                neuron = model.neurons[nid]
                writer.writerow([
                    nid,
                    neuron.position[0], neuron.position[1],
                    neuron.position[2], neuron.position[3],
                    neuron.area,
                    neuron.v_membrane, neuron.health, neuron.age,
                    neuron.last_spike_time
                ])
        
        print(f"Neurons exported to {output_path}")
    
    @staticmethod
    def export_synapses(model: BrainModel, output_path: Union[str, Path]) -> None:
        """Export synapse properties to CSV.
        
        Args:
            model: Brain model to export
            output_path: Output file path (.csv)
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'pre_id', 'post_id', 'weight', 'delay',
                'pre_area', 'post_area', 'distance'
            ])
            
            # Write synapse data
            for synapse in model.synapses:
                pre_neuron = model.neurons.get(synapse.pre_id)
                post_neuron = model.neurons.get(synapse.post_id)
                
                if pre_neuron and post_neuron:
                    distance = np.linalg.norm(
                        np.array(pre_neuron.position) - np.array(post_neuron.position)
                    )
                    
                    writer.writerow([
                        synapse.pre_id, synapse.post_id,
                        synapse.weight, synapse.delay,
                        pre_neuron.area, post_neuron.area,
                        distance
                    ])
        
        print(f"Synapses exported to {output_path}")
    
    @staticmethod
    def export_spike_times(
        spike_history: Dict[int, List[int]],
        output_path: Union[str, Path],
        dt: float = 1.0
    ) -> None:
        """Export spike times to CSV.
        
        Args:
            spike_history: Dictionary mapping neuron_id to list of spike times
            output_path: Output file path (.csv)
            dt: Time step size in ms
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['neuron_id', 'spike_time_step', 'spike_time_ms'])
            
            # Write spike data
            for neuron_id in sorted(spike_history.keys()):
                for spike_time in spike_history[neuron_id]:
                    writer.writerow([
                        neuron_id,
                        spike_time,
                        spike_time * dt
                    ])
        
        print(f"Spike times exported to {output_path}")
    
    @staticmethod
    def export_activity_summary(
        spike_history: Dict[int, List[int]],
        model: BrainModel,
        output_path: Union[str, Path]
    ) -> None:
        """Export per-neuron activity summary to CSV.
        
        Args:
            spike_history: Dictionary mapping neuron_id to list of spike times
            model: Brain model
            output_path: Output file path (.csv)
        """
        output_path = Path(output_path)
        
        # Calculate max time
        all_times = [t for times in spike_history.values() for t in times]
        max_time = max(all_times) if all_times else 1
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'neuron_id', 'area', 'n_spikes', 'firing_rate_hz',
                'mean_isi_ms', 'cv_isi', 'n_incoming', 'n_outgoing'
            ])
            
            # Write activity summary for each neuron
            for nid in sorted(model.neurons.keys()):
                neuron = model.neurons[nid]
                spikes = spike_history.get(nid, [])
                
                # Calculate firing rate
                firing_rate = len(spikes) / (max_time / 1000.0) if max_time > 0 else 0
                
                # Calculate ISI statistics
                if len(spikes) > 1:
                    isis = np.diff(sorted(spikes))
                    mean_isi = np.mean(isis)
                    cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else 0
                else:
                    mean_isi = 0
                    cv_isi = 0
                
                # Get connectivity
                n_incoming = len(model.get_synapses_for_neuron(nid, 'post'))
                n_outgoing = len(model.get_synapses_for_neuron(nid, 'pre'))
                
                writer.writerow([
                    nid, neuron.area, len(spikes), firing_rate,
                    mean_isi, cv_isi, n_incoming, n_outgoing
                ])
        
        print(f"Activity summary exported to {output_path}")


class MATLABExporter:
    """Export simulation data to MATLAB .mat format."""
    
    @staticmethod
    def export_all(
        model: BrainModel,
        spike_history: Dict[int, List[int]],
        output_path: Union[str, Path]
    ) -> None:
        """Export complete simulation data to MATLAB format.
        
        Requires scipy to be installed.
        
        Args:
            model: Brain model to export
            spike_history: Spike history dictionary
            output_path: Output file path (.mat)
        """
        try:
            from scipy.io import savemat
        except ImportError:
            print("Error: scipy is required for MATLAB export")
            print("Install with: pip install scipy")
            return
        
        output_path = Path(output_path)
        
        # Prepare neuron data
        neuron_ids = sorted(model.neurons.keys())
        neurons_struct = {
            'ids': np.array(neuron_ids),
            'positions': np.array([model.neurons[nid].position for nid in neuron_ids]),
            'v_membrane': np.array([model.neurons[nid].v_membrane for nid in neuron_ids]),
            'health': np.array([model.neurons[nid].health for nid in neuron_ids]),
            'age': np.array([model.neurons[nid].age for nid in neuron_ids]),
            'areas': [model.neurons[nid].area for nid in neuron_ids],  # Cell array in MATLAB
        }
        
        # Prepare synapse data
        synapses_struct = {
            'pre_ids': np.array([s.pre_id for s in model.synapses]),
            'post_ids': np.array([s.post_id for s in model.synapses]),
            'weights': np.array([s.weight for s in model.synapses]),
            'delays': np.array([s.delay for s in model.synapses]),
        }
        
        # Prepare spike data
        # Create spike matrix for easier MATLAB analysis
        all_times = [t for times in spike_history.values() for t in times]
        max_time = max(all_times) if all_times else 0
        n_neurons = len(neuron_ids)
        n_timesteps = max_time + 1
        spike_matrix = np.zeros((n_neurons, n_timesteps), dtype=np.uint8)
        
        for i, nid in enumerate(neuron_ids):
            for spike_time in spike_history.get(nid, []):
                if 0 <= spike_time < n_timesteps:
                    spike_matrix[i, spike_time] = 1
        
        spikes_struct = {
            'matrix': spike_matrix,
            'neuron_ids': np.array(neuron_ids),
            'n_timesteps': n_timesteps,
        }
        
        # Combine all data
        data_dict = {
            'neurons': neurons_struct,
            'synapses': synapses_struct,
            'spikes': spikes_struct,
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'n_neurons': len(neuron_ids),
                'n_synapses': len(model.synapses),
                'lattice_size': model.lattice_size,
            }
        }
        
        # Save to .mat file
        savemat(output_path, data_dict, oned_as='column')
        
        print(f"Data exported to MATLAB format: {output_path}")
        print(f"  Neurons: {len(neuron_ids)}")
        print(f"  Synapses: {len(model.synapses)}")
        print(f"  Timesteps: {n_timesteps}")


class NWBExporter:
    """Export simulation data to Neurodata Without Borders (NWB) format."""
    
    @staticmethod
    def export_to_nwb(
        model: BrainModel,
        spike_history: Dict[int, List[int]],
        output_path: Union[str, Path],
        session_description: str = "4D Neural Cognition Simulation",
        experimenter: str = "Simulation",
        dt: float = 0.001  # Time step in seconds
    ) -> None:
        """Export simulation to NWB 2.0 format.
        
        Requires pynwb to be installed.
        
        Args:
            model: Brain model to export
            spike_history: Spike history dictionary
            output_path: Output file path (.nwb)
            session_description: Description of the session
            experimenter: Name of experimenter
            dt: Time step size in seconds
        """
        try:
            from pynwb import NWBFile, TimeSeries, NWBHDF5IO
            from pynwb.ecephys import ElectricalSeries
            from datetime import datetime
            from dateutil.tz import tzlocal
        except ImportError:
            print("Error: pynwb is required for NWB export")
            print("Install with: pip install pynwb")
            return
        
        output_path = Path(output_path)
        
        # Create NWB file
        nwbfile = NWBFile(
            session_description=session_description,
            identifier=str(datetime.now().timestamp()),
            session_start_time=datetime.now(tzlocal()),
            experimenter=experimenter,
            lab="4D Neural Cognition",
            institution="Virtual Lab"
        )
        
        # Add neuron metadata as custom table
        # In NWB, we'd typically use Units table for this
        neuron_ids = sorted(model.neurons.keys())
        
        # Add spike times for each neuron
        for nid in neuron_ids:
            spikes = spike_history.get(nid, [])
            spike_times = np.array(spikes) * dt  # Convert to seconds
            
            neuron = model.neurons[nid]
            nwbfile.add_unit(
                spike_times=spike_times,
                id=nid,
                # Custom properties
                obs_intervals=None,
                electrodes=None,
                # Would add more metadata here in real use case
            )
        
        # Add network structure as processing module
        network_module = nwbfile.create_processing_module(
            name='network_structure',
            description='Network connectivity and structure'
        )
        
        # Store connectivity as time series (not ideal, but NWB is designed for ephys)
        # In practice, would use custom extension for network structure
        
        # Write to file
        with NWBHDF5IO(str(output_path), 'w') as io:
            io.write(nwbfile)
        
        print(f"Data exported to NWB format: {output_path}")
        print(f"  Neurons with spike data: {len(neuron_ids)}")
        print("  Note: NWB is primarily designed for experimental data.")
        print("  Network structure export is limited in standard NWB.")


def export_simulation(
    model: BrainModel,
    spike_history: Dict[int, List[int]],
    output_dir: Union[str, Path],
    formats: List[str] = None,
    prefix: str = "simulation"
) -> None:
    """Export simulation data to multiple formats at once.
    
    Args:
        model: Brain model to export
        spike_history: Spike history dictionary  
        output_dir: Output directory
        formats: List of formats to export ('numpy', 'csv', 'matlab', 'nwb')
                If None, exports to numpy and csv
        prefix: Prefix for output filenames
    """
    if formats is None:
        formats = ['numpy', 'csv']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting simulation data to {output_dir}")
    print(f"Formats: {', '.join(formats)}\n")
    
    # NumPy export
    if 'numpy' in formats:
        print("Exporting to NumPy format...")
        NumpyExporter.export_network_structure(
            model, output_dir / f"{prefix}_network.npz"
        )
        NumpyExporter.export_spike_trains(
            spike_history, output_dir / f"{prefix}_spikes.npz"
        )
    
    # CSV export
    if 'csv' in formats:
        print("\nExporting to CSV format...")
        CSVExporter.export_neurons(
            model, output_dir / f"{prefix}_neurons.csv"
        )
        CSVExporter.export_synapses(
            model, output_dir / f"{prefix}_synapses.csv"
        )
        CSVExporter.export_spike_times(
            spike_history, output_dir / f"{prefix}_spike_times.csv"
        )
        CSVExporter.export_activity_summary(
            spike_history, model, output_dir / f"{prefix}_activity.csv"
        )
    
    # MATLAB export
    if 'matlab' in formats:
        print("\nExporting to MATLAB format...")
        MATLABExporter.export_all(
            model, spike_history, output_dir / f"{prefix}_data.mat"
        )
    
    # NWB export
    if 'nwb' in formats:
        print("\nExporting to NWB format...")
        NWBExporter.export_to_nwb(
            model, spike_history, output_dir / f"{prefix}_data.nwb"
        )
    
    print(f"\n✅ Export complete! Files saved to {output_dir}")
