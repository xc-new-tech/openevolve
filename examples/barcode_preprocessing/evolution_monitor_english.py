#!/usr/bin/env python3
"""
OpenEvolve Evolution Monitor - Real-time Evolution Process Visualization
Supports real-time log parsing, multi-dimensional performance charts, optimal program tracking, etc.
"""

import os
import re
import json
import time
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import threading
import signal
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    import numpy as np
    import pandas as pd
    import matplotlib.font_manager as fm
    
    # Use default English fonts to avoid any font issues
    plt.style.use('default')
    print("English font configuration applied")
        
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install matplotlib numpy pandas")
    sys.exit(1)

class EvolutionMonitor:
    def __init__(self, output_dir="openevolve_output", update_interval=2.0, max_points=100, export=False):
        self.output_dir = Path(output_dir)
        self.update_interval = update_interval
        self.max_points = max_points
        
        # Data storage
        self.evolution_data = deque(maxlen=max_points)
        self.current_best = {'score': 0, 'iteration': 0, 'success_rate': 0.0}
        self.barcode_stats = defaultdict(lambda: deque(maxlen=max_points))
        
        self.start_time = None
        self.current_throughput = 0.0
        self.island_stats = {}
        
        self.running = False
        self.last_file_pos = 0
        self.log_file_path = None
        
        # Performance statistics
        self.performance_stats = {
            'total_evaluations': 0,
            'avg_eval_time': 0,
            'peak_score': 0,
            'peak_iteration': 0,
            'start_time': None,
            'current_iteration': 0
        }
        
        # Setup graphics interface
        self.setup_plots()
        
    def setup_plots(self):
        """Setup matplotlib graphics interface"""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('OpenEvolve Barcode Preprocessing Evolution Monitor', fontsize=16)
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main performance charts
        self.ax_score = self.fig.add_subplot(gs[0, :2])
        self.ax_success = self.fig.add_subplot(gs[1, 0])
        self.ax_time = self.fig.add_subplot(gs[1, 1])
        self.ax_barcode = self.fig.add_subplot(gs[2, 0])
        self.ax_island = self.fig.add_subplot(gs[2, 1])
        self.ax_stats = self.fig.add_subplot(gs[:, 2])
        
        # Configure charts
        self.setup_plot_styles()
        
    def setup_plot_styles(self):
        """Configure styles and labels for each chart"""
        # Score trend chart
        self.ax_score.set_title('Evolution Score Trend')
        self.ax_score.set_xlabel('Iteration')
        self.ax_score.set_ylabel('Score')
        self.ax_score.grid(True, alpha=0.3)
        self.ax_score.legend(['Current Score', 'Historical Best'])
        
        # Success rate chart
        self.ax_success.set_title('Success Rate Changes')
        self.ax_success.set_xlabel('Iteration')
        self.ax_success.set_ylabel('Success Rate (%)')
        self.ax_success.grid(True, alpha=0.3)
        
        # Execution time chart
        self.ax_time.set_title('Execution Time')
        self.ax_time.set_xlabel('Iteration')
        self.ax_time.set_ylabel('Time (seconds)')
        self.ax_time.grid(True, alpha=0.3)
        
        # Barcode type success rate
        self.ax_barcode.set_title('Barcode Type Success Rate')
        self.ax_barcode.set_xlabel('Iteration')
        self.ax_barcode.set_ylabel('Success Rate (%)')
        self.ax_barcode.grid(True, alpha=0.3)
        
        # Island status
        self.ax_island.set_title('Island Evolution Status')
        self.ax_island.set_xlabel('Island ID')
        self.ax_island.set_ylabel('Best Score')
        
        # Statistics panel
        self.ax_stats.set_title('Real-time Statistics')
        self.ax_stats.axis('off')
        
    def find_latest_log(self):
        """Find the latest log file"""
        logs_dir = self.output_dir / "logs"
        if not logs_dir.exists():
            return None
            
        log_files = list(logs_dir.glob("openevolve_*.log"))
        if not log_files:
            return None
            
        # Return the latest log file
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        return latest_log
        
    def parse_log_entry(self, line):
        """Parse log entry and extract key information"""
        try:
            # Parse iteration information
            if "Iteration" in line and "Child" in line:
                # Example: Iteration 5: Child abc123... score=90.00
                match = re.search(r'Iteration (\d+):.*score=([0-9.]+)', line)
                if match:
                    iteration = int(match.group(1))
                    score = float(match.group(2))
                    
                    # Extract more metrics
                    success_rate_match = re.search(r'success_rate=([0-9.]+)', line)
                    exec_time_match = re.search(r'execution_time=([0-9.]+)', line)
                    throughput_match = re.search(r'processed_throughput=([0-9.]+)', line)
                    
                    return {
                        'type': 'iteration',
                        'iteration': iteration,
                        'score': score,
                        'success_rate': float(success_rate_match.group(1)) * 100 if success_rate_match else 0,
                        'execution_time': float(exec_time_match.group(1)) if exec_time_match else 0,
                        'throughput': float(throughput_match.group(1)) if throughput_match else 0,
                        'timestamp': datetime.now()
                    }
                    
            # Parse island status
            elif "Island" in line and "programs" in line:
                # Example: Island 0: 5 programs, best=95.67, avg=80.23, diversity=0.45
                match = re.search(r'Island (\d+): (\d+) programs, best=([0-9.]+), avg=([0-9.]+), diversity=([0-9.]+)', line)
                if match:
                    return {
                        'type': 'island',
                        'island_id': int(match.group(1)),
                        'programs': int(match.group(2)),
                        'best_score': float(match.group(3)),
                        'avg_score': float(match.group(4)),
                        'diversity': float(match.group(5))
                    }
                    
            # Parse barcode type statistics
            elif re.search(r'(CODE128|CODE39|QRCODE): \d+/\d+ \([0-9.]+%\)', line):
                barcode_stats = {}
                matches = re.findall(r'(CODE128|CODE39|QRCODE): \d+/\d+ \(([0-9.]+)%\)', line)
                for barcode_type, success_rate in matches:
                    barcode_stats[barcode_type] = float(success_rate)
                if barcode_stats:
                    return {
                        'type': 'barcode_stats',
                        'stats': barcode_stats
                    }
                    
        except (ValueError, AttributeError) as e:
            pass  # Ignore parse errors
            
        return None
        
    def update_data(self, entry):
        """Update internal data structures"""
        if entry['type'] == 'iteration':
            # Update evolution data
            self.evolution_data.append({
                'iteration': entry['iteration'],
                'score': entry['score'],
                'success_rate': entry['success_rate'],
                'execution_time': entry['execution_time'],
                'throughput': entry['throughput'],
                'timestamp': entry['timestamp']
            })
            
            # Update current best
            if entry['score'] > self.current_best['score']:
                self.current_best = {
                    'score': entry['score'],
                    'iteration': entry['iteration'],
                    'success_rate': entry['success_rate']
                }
                
            # Update performance statistics
            self.performance_stats['current_iteration'] = entry['iteration']
            self.performance_stats['total_evaluations'] += 1
            if entry['score'] > self.performance_stats['peak_score']:
                self.performance_stats['peak_score'] = entry['score']
                self.performance_stats['peak_iteration'] = entry['iteration']
                
        elif entry['type'] == 'island':
            # Update island status
            self.island_stats[entry['island_id']] = entry
            
        elif entry['type'] == 'barcode_stats':
            # Update barcode statistics
            if self.evolution_data:
                current_iteration = self.evolution_data[-1]['iteration']
                for barcode_type, success_rate in entry['stats'].items():
                    self.barcode_stats[barcode_type].append({
                        'iteration': current_iteration,
                        'success_rate': success_rate
                    })
                    
    def read_log_updates(self):
        """Read new content from log file"""
        if not self.log_file_path or not self.log_file_path.exists():
            return []
            
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                f.seek(self.last_file_pos)
                new_content = f.read()
                self.last_file_pos = f.tell()
                
            # Parse new content
            new_entries = []
            for line in new_content.split('\n'):
                if line.strip():
                    entry = self.parse_log_entry(line)
                    if entry:
                        new_entries.append(entry)
                        
            return new_entries
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []
            
    def update_plots(self, frame):
        """Update all charts"""
        # Read new log data
        new_entries = self.read_log_updates()
        for entry in new_entries:
            self.update_data(entry)
            
        if not self.evolution_data:
            return
            
        # Extract data for plotting
        iterations = [d['iteration'] for d in self.evolution_data]
        scores = [d['score'] for d in self.evolution_data]
        success_rates = [d['success_rate'] for d in self.evolution_data]
        exec_times = [d['execution_time'] for d in self.evolution_data]
        
        # Update score trend chart
        self.ax_score.clear()
        self.ax_score.plot(iterations, scores, 'b-', label='Current Score', linewidth=2)
        if self.current_best['score'] > 0:
            self.ax_score.axhline(y=self.current_best['score'], color='r', linestyle='--', 
                                 label=f'Historical Best: {self.current_best["score"]:.2f}')
        self.ax_score.set_title('Evolution Score Trend')
        self.ax_score.set_xlabel('Iteration')
        self.ax_score.set_ylabel('Score')
        self.ax_score.grid(True, alpha=0.3)
        self.ax_score.legend()
        
        # Update success rate chart
        self.ax_success.clear()
        self.ax_success.plot(iterations, success_rates, 'g-', linewidth=2)
        self.ax_success.set_title('Success Rate Changes')
        self.ax_success.set_xlabel('Iteration')
        self.ax_success.set_ylabel('Success Rate (%)')
        self.ax_success.grid(True, alpha=0.3)
        
        # Update execution time chart
        self.ax_time.clear()
        self.ax_time.plot(iterations, exec_times, 'orange', linewidth=2)
        self.ax_time.set_title('Execution Time')
        self.ax_time.set_xlabel('Iteration')
        self.ax_time.set_ylabel('Time (seconds)')
        self.ax_time.grid(True, alpha=0.3)
        
        # Update barcode type success rate
        self.ax_barcode.clear()
        colors = ['red', 'green', 'blue']
        for i, (barcode_type, data) in enumerate(self.barcode_stats.items()):
            if data:
                barcode_iterations = [d['iteration'] for d in data]
                barcode_rates = [d['success_rate'] for d in data]
                self.ax_barcode.plot(barcode_iterations, barcode_rates, 
                                   color=colors[i % len(colors)], linewidth=2, 
                                   label=barcode_type)
        self.ax_barcode.set_title('Barcode Type Success Rate')
        self.ax_barcode.set_xlabel('Iteration')
        self.ax_barcode.set_ylabel('Success Rate (%)')
        self.ax_barcode.grid(True, alpha=0.3)
        self.ax_barcode.legend()
        
        # Update island status
        self.ax_island.clear()
        if self.island_stats:
            island_ids = list(self.island_stats.keys())
            best_scores = [self.island_stats[id]['best_score'] for id in island_ids]
            colors = plt.cm.viridis(np.linspace(0, 1, len(island_ids)))
            bars = self.ax_island.bar(island_ids, best_scores, color=colors)
            self.ax_island.set_title('Island Evolution Status')
            self.ax_island.set_xlabel('Island ID')
            self.ax_island.set_ylabel('Best Score')
            
            # Add value labels
            for bar, score in zip(bars, best_scores):
                height = bar.get_height()
                self.ax_island.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{score:.1f}', ha='center', va='bottom')
        
        # Update statistics panel
        self.update_stats_panel()
        
    def update_stats_panel(self):
        """Update statistics information panel"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Prepare statistics text
        stats_text = []
        
        # Basic statistics
        if self.evolution_data:
            current_data = self.evolution_data[-1]
            stats_text.extend([
                "=== Real-time Monitoring ===",
                f"Current Iteration: {current_data['iteration']}",
                f"Current Score: {current_data['score']:.2f}",
                f"Current Success Rate: {current_data['success_rate']:.1f}%",
                ""
            ])
        
        # Best records
        if self.current_best['score'] > 0:
            stats_text.extend([
                "=== Best Records ===",
                f"Highest Score: {self.current_best['score']:.2f}",
                f"Best Iteration: {self.current_best['iteration']}",
                f"Corresponding Success Rate: {self.current_best['success_rate']:.1f}%",
                ""
            ])
        
        # Runtime statistics
        if self.start_time:
            runtime = datetime.now() - self.start_time
            total_seconds = runtime.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            stats_text.extend([
                "=== Runtime Statistics ===",
                f"Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}",
                f"Total Evaluations: {self.performance_stats['total_evaluations']}",
                ""
            ])
        
        # Island summary
        if self.island_stats:
            stats_text.extend([
                "=== Island Status ===",
                f"Active Islands: {len(self.island_stats)}"
            ])
            for island_id, stats in self.island_stats.items():
                stats_text.append(f"Island {island_id}: {stats['best_score']:.1f}")
        
        # Display text
        full_text = '\n'.join(stats_text)
        self.ax_stats.text(0.05, 0.95, full_text, 
                          transform=self.ax_stats.transAxes,
                          verticalalignment='top',
                          fontsize=10,
                          family='monospace')
        
    def start_monitoring(self):
        """Start monitoring"""
        print(f"Starting OpenEvolve evolution monitoring...")
        print(f"Monitor directory: {self.output_dir}")
        print(f"Update interval: {self.update_interval} seconds")
        
        # Find log file
        self.log_file_path = self.find_latest_log()
        if not self.log_file_path:
            print("Log file not found, please ensure OpenEvolve is running")
            return
            
        print(f"Found log file: {self.log_file_path}")
        
        # Record start time
        self.start_time = datetime.now()
        self.performance_stats['start_time'] = self.start_time
        self.running = True
        
        # Setup close event handler
        def on_close(event):
            self.stop_monitoring()
            
        self.fig.canvas.mpl_connect('close_event', on_close)
        
        # Start animation
        ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                    interval=int(self.update_interval * 1000),
                                    blit=False)
        
        plt.tight_layout()
        plt.show()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        print("Monitoring stopped")
        
    def export_data(self, filename=None):
        """Export monitoring data to CSV"""
        if not self.evolution_data:
            print("No data to export")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evolution_data_{timestamp}.csv"
            
        # Convert to DataFrame and export
        df = pd.DataFrame(list(self.evolution_data))
        df.to_csv(filename, index=False)
        print(f"Data exported to: {filename}")
        
        # Export best records
        best_filename = filename.replace('.csv', '_best.json')
        with open(best_filename, 'w', encoding='utf-8') as f:
            json.dump(self.current_best, f, indent=2, ensure_ascii=False)
        print(f"Best records exported to: {best_filename}")

# Global monitor instance
monitor = None

def signal_handler(signum, frame):
    """Signal handler function"""
    if monitor:
        monitor.stop_monitoring()
    sys.exit(0)

def main():
    global monitor
    
    parser = argparse.ArgumentParser(description='OpenEvolve evolution process real-time monitoring')
    parser.add_argument('--output-dir', '-o', default='openevolve_output', 
                       help='OpenEvolve output directory (default: openevolve_output)')
    parser.add_argument('--interval', '-i', type=float, default=2.0,
                       help='Update interval in seconds (default: 2.0)')
    parser.add_argument('--max-points', '-m', type=int, default=100,
                       help='Maximum number of data points to display (default: 100)')
    parser.add_argument('--export', '-e', action='store_true',
                       help='Automatically export data when monitoring ends')
    
    args = parser.parse_args()
    
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start monitoring
    monitor = EvolutionMonitor(
        output_dir=args.output_dir,
        update_interval=args.interval,
        max_points=args.max_points,
        export=args.export
    )
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, stopping monitoring...")
    finally:
        if monitor and args.export:
            monitor.export_data()

if __name__ == "__main__":
    main() 