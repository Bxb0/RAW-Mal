"""
Training callbacks for Stable Baselines3.

This module provides custom callbacks for monitoring and logging
training progress.
"""

from typing import List, Optional
from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateCallback(BaseCallback):
    """
    Callback for tracking bypass success rate during training.
    
    Records cumulative and windowed success rates, and can plot
    training curves when matplotlib is available.
    """
    
    def __init__(self, verbose: int = 0, log_interval: int = 1000):
        """
        Initialize the callback.
        
        Args:
            verbose: Verbosity level.
            log_interval: Number of timesteps between log outputs.
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        
        # Cumulative statistics
        self.successes: int = 0
        self.total_episodes: int = 0
        
        # Window statistics (for recent success rate)
        self.window_successes: int = 0
        self.window_episodes: int = 0
        
        # Training curve data
        self.timesteps_log: List[int] = []
        self.success_rate_log: List[float] = []
        self.window_rate_log: List[float] = []
        self.episodes_log: List[int] = []
        
    def _on_step(self) -> bool:
        """
        Called at each training step.
        
        Returns:
            True to continue training, False to stop.
        """
        # Check for episode completion
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    self.total_episodes += 1
                    self.window_episodes += 1
                    infos = self.locals.get('infos', [])
                    if i < len(infos) and infos[i].get('bypassed', False):
                        self.successes += 1
                        self.window_successes += 1
        
        # Periodic logging
        if self.num_timesteps % self.log_interval == 0 and self.total_episodes > 0:
            # Cumulative success rate
            cumulative_rate = self.successes / self.total_episodes * 100
            
            # Window success rate (recent performance)
            window_rate = 0.0
            if self.window_episodes > 0:
                window_rate = self.window_successes / self.window_episodes * 100
            
            # Record data
            self.timesteps_log.append(self.num_timesteps)
            self.success_rate_log.append(cumulative_rate)
            self.window_rate_log.append(window_rate)
            self.episodes_log.append(self.total_episodes)
            
            print(f"[Step {self.num_timesteps}] Episodes: {self.total_episodes}, "
                  f"Success Rate: {cumulative_rate:.2f}% (cumulative), "
                  f"{window_rate:.2f}% (recent)")
            
            # Reset window statistics
            self.window_successes = 0
            self.window_episodes = 0
        
        return True
    
    def plot_training_curve(
        self, 
        save_path: str = "training_curve.png", 
        av_name: str = ""
    ) -> None:
        """
        Plot and save the training curve.
        
        Args:
            save_path: Path to save the plot.
            av_name: Name of the AV target (for title).
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(self.timesteps_log) < 2:
                print("Not enough data points to plot curve")
                return
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Success rate curve
            ax1.plot(self.timesteps_log, self.success_rate_log, 'b-', 
                    label='Cumulative Success Rate', linewidth=2)
            ax1.plot(self.timesteps_log, self.window_rate_log, 'r-', 
                    alpha=0.7, label='Recent Success Rate', linewidth=1)
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Success Rate (%)')
            ax1.set_title(f'Training Curve - {av_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Episode count curve
            ax2.plot(self.timesteps_log, self.episodes_log, 'g-', linewidth=2)
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Total Episodes')
            ax2.set_title('Episode Count')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            print(f"Training curve saved: {save_path}")
            
            # Print analysis
            self._analyze_curve()
            
        except ImportError:
            print("Warning: matplotlib not installed, cannot plot curve")
            print("Run: pip install matplotlib")
    
    def _analyze_curve(self) -> None:
        """Analyze training curve and provide recommendations."""
        if len(self.success_rate_log) < 5:
            return
        
        # Calculate trend from recent points
        recent_rates = self.success_rate_log[-5:]
        trend = recent_rates[-1] - recent_rates[0]
        
        print(f"\nTraining Analysis:")
        print(f"  Final success rate: {self.success_rate_log[-1]:.2f}%")
        print(f"  Recent trend: {'+' if trend > 0 else ''}{trend:.2f}%")
        
        if trend > 2:
            print(f"  Suggestion: Success rate still rising, consider more training")
        elif trend > -1:
            print(f"  Suggestion: Success rate stabilized, training can be stopped")
        else:
            print(f"  Suggestion: Success rate declining, may be overfitting")
    
    def get_final_stats(self) -> dict:
        """
        Get final training statistics.
        
        Returns:
            Dictionary with total_episodes, successes, and success_rate.
        """
        rate = 0.0
        if self.total_episodes > 0:
            rate = self.successes / self.total_episodes * 100
        
        return {
            'total_episodes': self.total_episodes,
            'successes': self.successes,
            'success_rate': rate
        }
