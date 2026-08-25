#!/usr/bin/env python3
"""
Universal Kernel K₂ Analysis for Nonlinear RG Flow Response

This script analyzes the universal nonlinear response kernel:
K₂(ω,φ₀) = lim_{ε→0} I_nonadd(ε,ω,φ₀)/ε²

The kernel characterizes the frequency-dependent nonlinear susceptibility
of the coupled RG flow system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, signal
from pathlib import Path
import json


class KernelK2Analyzer:
    """
    Analyze the universal kernel K₂(ω,φ₀) for perturbed RG flows
    """
    
    def __init__(self, epsilon_range, frequency_range=None):
        """
        Initialize kernel analyzer
        
        Parameters:
        -----------
        epsilon_range : array-like
            Range of perturbation amplitudes to analyze
        frequency_range : array-like, optional
            Range of frequencies ω for frequency-dependent analysis
        """
        self.epsilon = np.asarray(epsilon_range)
        self.frequencies = frequency_range
        self.K2_results = {}
        self.frequency_response = {}
        
    def compute_K2_from_single_frequency(self, epsilon_values, I_values):
        """
        Compute K₂ kernel for a single frequency
        
        Parameters:
        -----------
        epsilon_values : array-like
            Perturbation amplitudes
        I_values : array-like
            Non-additive observable values
            
        Returns:
        --------
        K2 : float
            Universal kernel value
        K2_uncertainty : float
            Statistical uncertainty in K₂
        """
        epsilon = np.asarray(epsilon_values)
        I = np.asarray(I_values)
        
        # Compute K₂ for each ε value
        K2_values = I / (epsilon**2)
        
        # Use weighted average with weights = 1/ε (smaller ε = more reliable)
        weights = 1.0 / epsilon
        weights = weights / weights.sum()
        
        K2_weighted = np.average(K2_values, weights=weights)
        K2_std = np.sqrt(np.average((K2_values - K2_weighted)**2, weights=weights))
        
        return K2_weighted, K2_std
    
    def extrapolate_to_zero_epsilon(self, epsilon_values, I_values, method='polynomial'):
        """
        Extrapolate K₂ to ε → 0 limit
        
        Parameters:
        -----------
        epsilon_values : array-like
            Perturbation amplitudes
        I_values : array-like
            Non-additive observable values
        method : str
            Extrapolation method ('polynomial', 'rational', 'exponential')
            
        Returns:
        --------
        K2_limit : float
            K₂ in the limit ε → 0
        extrapolation_error : float
            Uncertainty in extrapolation
        """
        epsilon = np.asarray(epsilon_values)
        I = np.asarray(I_values)
        
        K2_values = I / (epsilon**2)
        
        if method == 'polynomial':
            # Fit polynomial in ε to extrapolate to ε=0
            def poly_model(eps, a0, a1, a2):
                return a0 + a1*eps + a2*eps**2
            
            try:
                params, _ = optimize.curve_fit(poly_model, epsilon, K2_values)
                K2_limit = params[0]  # a0 is the ε→0 limit
                extrapolation_error = np.sqrt(np.diag(optimize.curve_fit(poly_model, epsilon, K2_values)[1]))[0]
            except:
                # Fallback to simple average
                K2_limit = np.mean(K2_values)
                extrapolation_error = np.std(K2_values)
                
        elif method == 'rational':
            # Fit rational function: K₂(ε) = (a + bε) / (1 + cε)
            def rational_model(eps, a, b, c):
                return (a + b*eps) / (1 + c*eps)
            
            try:
                params, _ = optimize.curve_fit(rational_model, epsilon, K2_values)
                K2_limit = params[0]  # a is the ε→0 limit
                extrapolation_error = np.sqrt(np.diag(optimize.curve_fit(rational_model, epsilon, K2_values)[1]))[0]
            except:
                K2_limit = np.mean(K2_values)
                extrapolation_error = np.std(K2_values)
                
        else:  # exponential
            # Fit exponential approach: K₂(ε) = K₂∞ + A exp(-ε/λ)
            def exp_model(eps, K_inf, A, lam):
                return K_inf + A * np.exp(-eps/lam)
            
            try:
                params, _ = optimize.curve_fit(exp_model, epsilon, K2_values, 
                                               p0=[np.mean(K2_values), 0.01, 0.1])
                K2_limit = params[0]
                extrapolation_error = np.sqrt(np.diag(optimize.curve_fit(exp_model, epsilon, K2_values,
                                                                        p0=[np.mean(K2_values), 0.01, 0.1])[1]))[0]
            except:
                K2_limit = np.mean(K2_values)
                extrapolation_error = np.std(K2_values)
        
        return K2_limit, extrapolation_error
    
    def analyze_frequency_dependence(self, frequency_data):
        """
        Analyze K₂ as a function of frequency ω
        
        Parameters:
        -----------
        frequency_data : dict
            Dictionary mapping frequencies to (epsilon, I) data tuples
            {ω₁: (ε_array₁, I_array₁), ω₂: (ε_array₂, I_array₂), ...}
            
        Returns:
        --------
        frequency_response : dict
            K₂ values and uncertainties for each frequency
        """
        freq_response = {}
        
        for omega, (eps, I_vals) in frequency_data.items():
            K2, K2_err = self.extrapolate_to_zero_epsilon(eps, I_vals)
            freq_response[omega] = {
                'K2': K2,
                'K2_error': K2_err,
                'epsilon_data': eps,
                'I_data': I_vals
            }
        
        self.frequency_response = freq_response
        return freq_response
    
    def detect_resonance_features(self, frequencies, K2_values, K2_errors):
        """
        Detect potential resonance features in K₂(ω)
        
        Parameters:
        -----------
        frequencies : array-like
            Frequency values
        K2_values : array-like
            K₂ values at each frequency
        K2_errors : array-like
            Uncertainties in K₂ values
            
        Returns:
        --------
        resonances : list
            List of detected resonance frequencies
        """
        freqs = np.asarray(frequencies)
        K2 = np.asarray(K2_values)
        errors = np.asarray(K2_errors)
        
        # Smooth the data to find peaks
        if len(freqs) > 5:
            window_size = min(5, len(freqs) // 2)
            K2_smooth = np.convolve(K2, np.ones(window_size)/window_size, mode='valid')
            freq_smooth = freqs[window_size//2:-(window_size//2)+1]
            
            # Find peaks using signal processing
            peaks, _ = signal.find_peaks(K2_smooth, height=np.mean(K2_smooth))
            
            resonances = []
            for peak_idx in peaks:
                resonance_freq = freq_smooth[peak_idx]
                resonance_K2 = K2_smooth[peak_idx]
                resonances.append({
                    'frequency': resonance_freq,
                    'K2_value': resonance_K2,
                    'significance': resonance_K2 / np.mean(K2_smooth)
                })
        else:
            resonances = []
        
        return resonances
    
    def test_golden_ratio_hypothesis(self, frequencies, K2_values):
        """
        Test if K₂ shows enhancement at golden-ratio frequency
        
        Parameters:
        -----------
        frequencies : array-like
            Frequency values
        K2_values : array-like
            K₂ values at each frequency
            
        Returns:
        --------
        test_result : dict
            Results of golden-ratio hypothesis test
        """
        phi = (1 + np.sqrt(5)) / 2
        golden_freq = 2 * np.pi / np.log(phi)
        
        freqs = np.asarray(frequencies)
        K2 = np.asarray(K2_values)
        
        # Find K₂ at closest frequency to golden ratio
        idx = np.argmin(np.abs(freqs - golden_freq))
        closest_freq = freqs[idx]
        K2_at_golden = K2[idx]
        
        # Compare to average K₂
        K2_mean = np.mean(K2)
        enhancement_factor = K2_at_golden / K2_mean
        
        # Statistical significance (simple z-test)
        K2_std = np.std(K2)
        z_score = (K2_at_golden - K2_mean) / K2_std if K2_std > 0 else 0
        
        test_result = {
            'golden_ratio_frequency': golden_freq,
            'closest_measured_frequency': closest_freq,
            'K2_at_golden': K2_at_golden,
            'K2_mean': K2_mean,
            'enhancement_factor': enhancement_factor,
            'z_score': z_score,
            'is_significant': abs(z_score) > 2.0,  # 2σ threshold
            'conclusion': 'Golden ratio selection NOT statistically significant' if abs(z_score) < 2.0 
                          else 'Golden ratio selection may be significant'
        }
        
        return test_result
    
    def plot_K2_analysis(self, save_path=None):
        """
        Create comprehensive K₂ analysis plots
        
        Parameters:
        -----------
        save_path : str or Path, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: K₂ vs ε (if we have epsilon-dependent data)
        ax1 = axes[0, 0]
        if hasattr(self, 'K2_values'):
            ax1.loglog(self.epsilon, self.K2_values, 'bo-', markersize=8, label='K₂(ε)')
            ax1.axhline(y=self.K2_mean, color='r', linestyle='--', 
                       label=f'K₂ = {self.K2_mean:.6f}')
            ax1.set_xlabel('Perturbation Amplitude ε', fontsize=12)
            ax1.set_ylabel('K₂(ε) = I/ε²', fontsize=12)
            ax1.set_title('Universal Kernel vs Perturbation', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No ε-dependent data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Universal Kernel vs Perturbation', fontsize=14, fontweight='bold')
        
        # Plot 2: K₂ vs ω (frequency dependence)
        ax2 = axes[0, 1]
        if self.frequency_response:
            freqs = sorted(self.frequency_response.keys())
            K2_vals = [self.frequency_response[f]['K2'] for f in freqs]
            K2_errs = [self.frequency_response[f]['K2_error'] for f in freqs]
            
            ax2.errorbar(freqs, K2_vals, yerr=K2_errs, fmt='ro-', 
                        markersize=8, capsize=5, label='K₂(ω)')
            ax2.set_xlabel('Frequency ω', fontsize=12)
            ax2.set_ylabel('K₂(ω)', fontsize=12)
            ax2.set_title('Frequency Dependence of Universal Kernel', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Mark golden ratio frequency
            phi = (1 + np.sqrt(5)) / 2
            golden_freq = 2 * np.pi / np.log(phi)
            ax2.axvline(x=golden_freq, color='gold', linestyle='--', linewidth=2,
                       label=f'Golden ratio: ω = {golden_freq:.4f}')
            ax2.legend(fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No frequency-dependent data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Frequency Dependence of Universal Kernel', fontsize=14, fontweight='bold')
        
        # Plot 3: Extrapolation to ε→0
        ax3 = axes[1, 0]
        if hasattr(self, 'K2_values'):
            epsilon_fit = np.linspace(0, self.epsilon.max(), 100)
            
            # Show different extrapolation methods
            methods = ['polynomial', 'rational', 'exponential']
            colors = ['b', 'g', 'r']
            
            for method, color in zip(methods, colors):
                try:
                    def fit_func(eps, a0, a1, a2):
                        if method == 'polynomial':
                            return a0 + a1*eps + a2*eps**2
                        elif method == 'rational':
                            return (a0 + a1*eps) / (1 + a2*eps)
                        else:  # exponential
                            return a0 + a1*np.exp(-eps/a2)
                    
                    params, _ = optimize.curve_fit(fit_func, self.epsilon, self.K2_values)
                    K2_fit = fit_func(epsilon_fit, *params)
                    ax3.plot(epsilon_fit, K2_fit, color=color, linestyle='--', 
                            label=f'{method}: K₂(0) = {params[0]:.6f}')
                except:
                    pass
            
            ax3.scatter(self.epsilon, self.K2_values, color='k', s=50, zorder=5, label='Data')
            ax3.set_xlabel('Perturbation Amplitude ε', fontsize=12)
            ax3.set_ylabel('K₂(ε)', fontsize=12)
            ax3.set_title('Extrapolation to ε → 0', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No ε-dependent data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Extrapolation to ε → 0', fontsize=14, fontweight='bold')
        
        # Plot 4: Resonance analysis
        ax4 = axes[1, 1]
        if self.frequency_response:
            freqs = sorted(self.frequency_response.keys())
            K2_vals = [self.frequency_response[f]['K2'] for f in freqs]
            
            # Normalize to see relative variations
            K2_normalized = np.array(K2_vals) / np.mean(K2_vals)
            ax4.plot(freqs, K2_normalized, 'bo-', markersize=8)
            ax4.axhline(y=1.0, color='k', linestyle='-', linewidth=1)
            
            # Mark potential resonances
            if len(freqs) > 5:
                K2_smooth = np.convolve(K2_normalized, np.ones(3)/3, mode='valid')
                freq_smooth = freqs[1:-1]
                peaks, _ = signal.find_peaks(K2_smooth, height=1.0)
                for peak in peaks:
                    ax4.scatter(freq_smooth[peak], K2_smooth[peak], 
                              color='red', s=100, zorder=5, marker='*')
            
            ax4.set_xlabel('Frequency ω', fontsize=12)
            ax4.set_ylabel('Normalized K₂(ω)', fontsize=12)
            ax4.set_title('Resonance Feature Detection', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No frequency-dependent data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Resonance Feature Detection', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def generate_kernel_report(self):
        """
        Generate comprehensive kernel analysis report
        
        Returns:
        --------
        report : dict
            Dictionary containing all kernel analysis results
        """
        report = {
            'kernel_analysis': {
                'universal_kernel_K2': float(self.K2_mean) if hasattr(self, 'K2_mean') else None,
                'K2_uncertainty': float(self.K2_std) if hasattr(self, 'K2_std') else None,
                'extrapolation_methods_tested': ['polynomial', 'rational', 'exponential']
            },
            'frequency_analysis': self.frequency_response if self.frequency_response else None,
            'golden_ratio_test': None,
            'conclusion': self._generate_kernel_conclusion()
        }
        
        return report
    
    def _generate_kernel_conclusion(self):
        """Generate text conclusion for kernel analysis"""
        conclusion = """
UNIVERSAL KERNEL K₂ ANALYSIS RESULTS
====================================

The universal kernel K₂(ω,φ₀) characterizes the nonlinear susceptibility
of the coupled RG flow to weak perturbations.

Key findings:
- K₂ represents the leading-order response coefficient in the expansion:
  I_nonadd = ε² K₂(ω,φ₀) + O(ε³)
- The kernel is universal in the sense that it depends only on the 
  frequency and phase, not on the perturbation amplitude in the ε→0 limit.
- Frequency dependence of K₂ must be determined experimentally.

Status: Quadratic scaling (p ≈ 2) is well-established from numerical experiments.
The frequency dependence K₂(ω) and potential golden-ratio selection require
additional statistical validation.
"""
        return conclusion


def example_usage():
    """Example usage of KernelK2Analyzer"""
    print("=" * 60)
    print("UNIVERSAL KERNEL K₂ ANALYSIS")
    print("=" * 60)
    print()
    
    # Example data from Audit V4.2-V4.3
    epsilon = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    I_max = np.array([1.543259e-9, 1.388952e-8, 1.543240e-7, 
                      1.388820e-6, 1.542762e-5, 1.387539e-4, 1.538148e-3])
    
    # Create analyzer
    analyzer = KernelK2Analyzer(epsilon)
    
    # Compute K₂ values
    K2_values = I_max / (epsilon**2)
    analyzer.K2_values = K2_values
    analyzer.K2_mean = np.mean(K2_values)
    analyzer.K2_std = np.std(K2_values)
    
    print(f"Universal Kernel K₂ = {analyzer.K2_mean:.6f} ± {analyzer.K2_std:.6f}")
    print()
    
    # Extrapolate to ε→0 using different methods
    print("Extrapolation to ε → 0:")
    for method in ['polynomial', 'rational', 'exponential']:
        K2_limit, error = analyzer.extrapolate_to_zero_epsilon(epsilon, I_max, method=method)
        print(f"  {method.capitalize()}: K₂(0) = {K2_limit:.6f} ± {error:.6f}")
    print()
    
    # Generate report
    report = analyzer.generate_kernel_report()
    print(report['conclusion'])
    
    # Create plots
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "kernel_K2_analysis.png"
    analyzer.plot_K2_analysis(save_path=plot_path)
    
    # Save results
    import json
    report_path = output_dir / "kernel_K2_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    example_usage()