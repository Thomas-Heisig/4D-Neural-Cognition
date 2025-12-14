"""Learning Trajectory Analysis Tool.

Analyzes learning paths from autonomous learning experiments,
identifying phase transitions, emergent patterns, and developmental stages.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LearningTrajectoryAnalyzer:
    """Analyzer for learning trajectories.
    
    Identifies patterns in autonomous learning including:
    - Phase transitions (e.g., random exploration to goal-directed)
    - Strategy evolution
    - Learning progress over time
    - Emergent developmental stages
    
    Attributes:
        data: Loaded experiment data
        cycle_results: Per-cycle results
        metrics: Computed analysis metrics
    """
    
    def __init__(self, data: Dict):
        """Initialize analyzer with experiment data.
        
        Args:
            data: Loaded experiment results
        """
        self.data = data
        self.cycle_results = data.get('cycle_results', [])
        self.metrics: Dict = {}
        
        logger.info(
            f"Initialized LearningTrajectoryAnalyzer with "
            f"{len(self.cycle_results)} cycles"
        )
    
    def analyze(self) -> Dict:
        """Run complete analysis.
        
        Returns:
            Dictionary with analysis results
        """
        logger.info("Running learning trajectory analysis...")
        
        self.metrics = {
            'phase_transitions': self._detect_phase_transitions(),
            'strategy_evolution': self._analyze_strategy_evolution(),
            'learning_progress': self._measure_learning_progress(),
            'goal_distribution': self._analyze_goal_distribution(),
            'developmental_stages': self._identify_developmental_stages(),
        }
        
        logger.info("Analysis complete")
        return self.metrics
    
    def _detect_phase_transitions(self) -> List[Dict]:
        """Detect phase transitions in learning behavior.
        
        Returns:
            List of detected phase transitions
        """
        if not self.cycle_results:
            return []
        
        transitions = []
        window_size = 50
        
        # Analyze prediction error transitions
        pred_errors = [c.get('prediction_error', 0.5) for c in self.cycle_results]
        
        for i in range(window_size, len(pred_errors) - window_size):
            before_avg = np.mean(pred_errors[i-window_size:i])
            after_avg = np.mean(pred_errors[i:i+window_size])
            
            # Significant drop in prediction error indicates learning transition
            if before_avg > 0 and (before_avg - after_avg) / before_avg > 0.3:
                transitions.append({
                    'cycle': i,
                    'type': 'prediction_error_reduction',
                    'before_value': before_avg,
                    'after_value': after_avg,
                    'magnitude': (before_avg - after_avg) / before_avg,
                })
        
        # Analyze strategy transitions
        for i in range(1, len(self.cycle_results)):
            prev_strategy = self.cycle_results[i-1].get('strategy')
            curr_strategy = self.cycle_results[i].get('strategy')
            
            if prev_strategy and curr_strategy and prev_strategy != curr_strategy:
                transitions.append({
                    'cycle': i,
                    'type': 'strategy_change',
                    'from_strategy': prev_strategy,
                    'to_strategy': curr_strategy,
                })
        
        logger.info(f"Detected {len(transitions)} phase transitions")
        return transitions
    
    def _analyze_strategy_evolution(self) -> Dict:
        """Analyze how learning strategies evolved over time.
        
        Returns:
            Dictionary with strategy analysis
        """
        if not self.cycle_results:
            return {}
        
        strategies = [c.get('strategy', 'unknown') for c in self.cycle_results]
        
        # Count strategy usage
        strategy_counts = {}
        for s in strategies:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        
        # Calculate strategy durations
        strategy_durations = []
        current_strategy = strategies[0]
        duration = 1
        
        for s in strategies[1:]:
            if s == current_strategy:
                duration += 1
            else:
                strategy_durations.append({
                    'strategy': current_strategy,
                    'duration': duration,
                })
                current_strategy = s
                duration = 1
        
        strategy_durations.append({
            'strategy': current_strategy,
            'duration': duration,
        })
        
        return {
            'strategy_counts': strategy_counts,
            'strategy_durations': strategy_durations,
            'total_strategy_changes': sum(1 for d in strategy_durations if d['duration'] > 1),
            'dominant_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0],
        }
    
    def _measure_learning_progress(self) -> Dict:
        """Measure overall learning progress.
        
        Returns:
            Dictionary with learning metrics
        """
        if not self.cycle_results:
            return {}
        
        # Extract metrics over time
        pred_errors = [c.get('prediction_error', 0.5) for c in self.cycle_results]
        world_model_accuracies = [c.get('world_model_accuracy', 0.0) for c in self.cycle_results]
        discoveries = [c.get('total_discovered', 0) for c in self.cycle_results]
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            return coeffs[0]  # Slope
        
        return {
            'initial_prediction_error': pred_errors[0] if pred_errors else 0.0,
            'final_prediction_error': pred_errors[-1] if pred_errors else 0.0,
            'error_reduction': pred_errors[0] - pred_errors[-1] if len(pred_errors) > 1 else 0.0,
            'error_trend': calculate_trend(pred_errors),
            'world_model_accuracy_trend': calculate_trend(world_model_accuracies),
            'discovery_rate': discoveries[-1] / len(self.cycle_results) if self.cycle_results else 0.0,
            'total_discoveries': discoveries[-1] if discoveries else 0,
        }
    
    def _analyze_goal_distribution(self) -> Dict:
        """Analyze distribution of pursued goals.
        
        Returns:
            Dictionary with goal statistics
        """
        if not self.cycle_results:
            return {}
        
        goal_types = [c.get('goal_type', 'unknown') for c in self.cycle_results if c.get('goal_type')]
        
        # Count goal type frequencies
        goal_counts = {}
        for g in goal_types:
            goal_counts[g] = goal_counts.get(g, 0) + 1
        
        return {
            'goal_counts': goal_counts,
            'total_goals': len(goal_types),
            'unique_goal_types': len(goal_counts),
            'most_common_goal': max(goal_counts.items(), key=lambda x: x[1])[0] if goal_counts else None,
        }
    
    def _identify_developmental_stages(self) -> List[Dict]:
        """Identify emergent developmental stages.
        
        Similar to infant development stages (reflexive → exploratory → goal-directed).
        
        Returns:
            List of identified stages
        """
        if len(self.cycle_results) < 100:
            return []
        
        stages = []
        
        # Divide timeline into segments
        segment_size = len(self.cycle_results) // 5
        
        for i in range(0, len(self.cycle_results), segment_size):
            segment = self.cycle_results[i:i+segment_size]
            
            # Analyze characteristics of this segment
            strategies = [c.get('strategy', '') for c in segment]
            pred_errors = [c.get('prediction_error', 0.5) for c in segment]
            discoveries = [c.get('newly_discovered', 0) for c in segment]
            
            # Determine stage characteristics
            avg_pred_error = np.mean(pred_errors)
            avg_discoveries = np.mean(discoveries)
            dominant_strategy = max(set(strategies), key=strategies.count) if strategies else 'unknown'
            
            # Classify stage
            if avg_pred_error > 0.6 and dominant_strategy == 'explore':
                stage_name = "Random Exploration"
            elif avg_pred_error > 0.4 and avg_discoveries > 0.5:
                stage_name = "Active Discovery"
            elif dominant_strategy == 'exploit':
                stage_name = "Skill Refinement"
            elif dominant_strategy == 'consolidate':
                stage_name = "Mastery & Consolidation"
            else:
                stage_name = "Transitional"
            
            stages.append({
                'start_cycle': i,
                'end_cycle': i + segment_size,
                'stage_name': stage_name,
                'avg_prediction_error': avg_pred_error,
                'avg_discoveries': avg_discoveries,
                'dominant_strategy': dominant_strategy,
            })
        
        logger.info(f"Identified {len(stages)} developmental stages")
        return stages
    
    def plot_phase_transitions(self, output_file: Optional[str] = None) -> None:
        """Plot phase transitions (placeholder).
        
        Args:
            output_file: Optional file to save plot
        """
        logger.info("Plotting phase transitions...")
        
        # In a real implementation, this would use matplotlib
        # For now, just log the transitions
        transitions = self.metrics.get('phase_transitions', [])
        
        logger.info(f"Phase Transitions ({len(transitions)} total):")
        for t in transitions:
            logger.info(f"  Cycle {t['cycle']}: {t['type']}")
        
        if output_file:
            logger.info(f"Would save plot to {output_file}")
    
    def generate_report(self, output_file: str) -> None:
        """Generate analysis report.
        
        Args:
            output_file: File to save report
        """
        logger.info("Generating analysis report...")
        
        report = {
            'experiment_config': self.data.get('config', {}),
            'experiment_summary': {
                'total_cycles': len(self.cycle_results),
                'elapsed_time': self.data.get('elapsed_time_seconds', 0),
            },
            'analysis_results': self.metrics,
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze learning trajectory from autonomous learning experiment'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file with experiment results'
    )
    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        help='Plot type: phase_transitions, learning_curve, strategy_evolution'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for analysis report'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Create analyzer
    analyzer = LearningTrajectoryAnalyzer(data)
    
    # Run analysis
    metrics = analyzer.analyze()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("LEARNING TRAJECTORY ANALYSIS")
    logger.info("="*60)
    
    # Phase transitions
    transitions = metrics.get('phase_transitions', [])
    logger.info(f"Phase transitions detected: {len(transitions)}")
    
    # Strategy evolution
    strategy_evo = metrics.get('strategy_evolution', {})
    logger.info(f"Strategy changes: {strategy_evo.get('total_strategy_changes', 0)}")
    logger.info(f"Dominant strategy: {strategy_evo.get('dominant_strategy', 'unknown')}")
    
    # Learning progress
    progress = metrics.get('learning_progress', {})
    logger.info(f"Prediction error reduction: {progress.get('error_reduction', 0):.3f}")
    logger.info(f"Total discoveries: {progress.get('total_discoveries', 0)}")
    
    # Developmental stages
    stages = metrics.get('developmental_stages', [])
    logger.info(f"\nDevelopmental stages ({len(stages)}):")
    for stage in stages:
        logger.info(f"  Cycles {stage['start_cycle']}-{stage['end_cycle']}: {stage['stage_name']}")
    
    logger.info("="*60)
    
    # Generate plot if requested
    if args.plot:
        if args.plot == 'phase_transitions':
            analyzer.plot_phase_transitions(f"{args.input.replace('.json', '')}_transitions.png")
    
    # Generate report if requested
    if args.output:
        analyzer.generate_report(args.output)
    
    return metrics


if __name__ == '__main__':
    main()
