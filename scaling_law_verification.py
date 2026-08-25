#!/usr/bin/env python3
"""
Universal Nonlinear Scaling Law Verification for Perturbed RG Flows
Based on Audit V4.2-V4.3 results

This script verifies the quadratic scaling law:
I_nonadd(ε,ω,φ₀) = ε² K₂(ω,φ₀) + O(ε³)

where p ≈ 2.0 from numerical experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json


class ScalingLawVerification:
    """
    Verify the universal scaling law I ∝ ε^p with p ≈ 2
    """
    
    def __init__(self, epsilon_values, I_values):
        """
        Initialize with experimental data
        
        Parameters:
        -----------
        epsilon_values : array-like
            Perturbation amplitudes
        I_values : array-like  
            Non-additive observable values
        """
        self.epsilon = np.asarray(epsilon_values)
        self.I = np.asarray(I_values)
        self.p_fit = None
        self.K2 = None
        self拟合结果 = None
        
    def power_law(self, epsilon, p, C):
        """Power law model: I = C * ε^p"""
        return C * epsilon**p
    
    def fit_scaling_exponent(self):
        """
        Fit the scaling exponent p in I ∝ ε^p
        
        Returns:
        --------
        p : float
            Fitted scaling exponent (expected ≈ 2.0)
        C : float
            Amplitude constant
        """
        # Use log-linear fitting for stability
        log_epsilon = np.log(self.epsilon)
        log_I = np.log(self.I)
        
        # Linear fit: log(I) = log(C) + p * log(ε)
        coeffs = np.polyfit(log_epsilon, log_I, 1)
        p_fit = coeffs[0]
        log_C = coeffs[1]
        C_fit = np.exp(log_C)
        
        self.p_fit = p_fit
        self.K2 = C_fit  # K₂ is the amplitude constant
        
        # Store fitting results
        self.fitting_results = {
            'p': p_fit,
            'C': C_fit,
            'K2': C_fit,
            'theoretical_p': 2.0,
            'deviation_from_theory': abs(p_fit - 2.0)
        }
        
        return p_fit, C_fit
    
    def compute_K2_kernel(self):
        """
        Compute the universal kernel K₂(ω) = lim_{ε→0} I(ε,ω)/ε²
        
        Returns:
        --------
        K2_values : array
            K₂ computed for each ε value
        K2_mean : float
            Mean K₂ value (universal kernel)
        """
        K2_values = self.I / (self.epsilon**2)
        K2_mean = np.mean(K2_values)
        K2_std = np.std(K2_values)
        
        self.K2_values = K2_values
        self.K2_mean = K2_mean
        self.K2_std = K2_std
        
        return K2_values, K2_mean, K2_std
    
    def verify_quadratic_scaling(self, tolerance=0.01):
        """
        Verify if scaling is quadratic within tolerance
        
        Parameters:
        -----------
        tolerance : float
            Acceptable deviation from p=2
        
        Returns:
        --------
        is_quadratic : bool
            True if p ≈ 2 within tolerance
        """
        if self.p_fit is None:
            self.fit_scaling_exponent()
        
        deviation = abs(self.p_fit - 2.0)
        is_quadratic = deviation <= tolerance
        
        return is_quadratic, deviation
    
    def plot_scaling_analysis(self, save_path=None):
        """
        Create comprehensive scaling analysis plots
        
        Parameters:
        -----------
        save_path : str or Path, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Log-log scaling
        ax1 = axes[0, 0]
        ax1.loglog(self.epsilon, self.I, 'bo-', label='Data', markersize=8)
        
        if self.p_fit is not None:
            epsilon_fit = np.logspace(np.log10(self.epsilon.min()), 
                                     np.log10(self.epsilon.max()), 100)
            I_fit = self.K2 * epsilon_fit**self.p_fit
            ax1.loglog(epsilon_fit, I_fit, 'r--', 
                      label=f'Fit: p={self.p_fit:.4f}', linewidth=2)
            
            # Theoretical p=2 line
            I_theory = self.K2 * epsilon_fit**2
            ax1.loglog(epsilon_fit, I_theory, 'g:', 
                      label='Theory: p=2.0', linewidth=2)
        
        ax1.set_xlabel('Perturbation Amplitude ε', fontsize=12)
        ax1.set_ylabel('Non-additive Observable I', fontsize=12)
        ax1.set_title('Log-Log Scaling Analysis', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: K₂ kernel analysis
        ax2 = axes[0, 1]
        K2_vals = self.I / (self.epsilon**2)
        ax2.semilogx(self.epsilon, K2_vals, 'ro-', markersize=8)
        ax2.axhline(y=self.K2_mean, color='b', linestyle='--', 
                   label=f'K₂ = {self.K2_mean:.6f}')
        ax2.fill_between(self.epsilon, 
                         self.K2_mean - self.K2_std,
                         self.K2_mean + self.K2_std,
                         alpha=0.2, color='b', label='±1σ')
        ax2.set_xlabel('Perturbation Amplitude ε', fontsize=12)
        ax2.set_ylabel('K₂(ε) = I/ε²', fontsize=12)
        ax2.set_title('Universal Kernel K₂ Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Relative deviation from quadratic
        ax3 = axes[1, 0]
        I_theoretical = self.K2 * self.epsilon**2
        relative_error = (self.I - I_theoretical) / I_theoretical * 100
        ax3.semilogx(self.epsilon, relative_error, 'go-', markersize=8)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax3.set_xlabel('Perturbation Amplitude ε', fontsize=12)
        ax3.set_ylabel('Relative Deviation from Quadratic (%)', fontsize=12)
        ax3.set_title('Deviation from p=2 Scaling', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Residual analysis
        ax4 = axes[1, 1]
        if self.p_fit is not None:
            I_predicted = self.K2 * self.epsilon**self.p_fit
            residuals = self.I - I_predicted
            ax4.plot(self.epsilon, residuals, 'mo-', markersize=8)
            ax4.axhline(y=0, color='k', linestyle='-', linewidth=1)
            ax4.set_xlabel('Perturbation Amplitude ε', fontsize=12)
            ax4.set_ylabel('Residuals (I - I_fit)', fontsize=12)
            ax4.set_title('Residual Analysis', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def generate_report(self):
        """
        Generate a comprehensive analysis report
        
        Returns:
        --------
        report : dict
            Dictionary containing all analysis results
        """
        if self.p_fit is None:
            self.fit_scaling_exponent()
        
        K2_vals, K2_mean, K2_std = self.compute_K2_kernel()
        is_quadratic, deviation = self.verify_quadratic_scaling()
        
        report = {
            'scaling_analysis': {
                'fitted_exponent_p': float(self.p_fit),
                'theoretical_exponent': 2.0,
                'deviation_from_theory': float(deviation),
                'is_quadratic_scaling': bool(is_quadratic),
                'amplitude_constant_C': float(self.K2)
            },
            'kernel_analysis': {
                'K2_mean': float(K2_mean),
                'K2_std': float(K2_std),
                'K2_values': [float(v) for v in K2_vals],
                'K2_relative_std': float(K2_std / K2_mean * 100)
            },
            'data_summary': {
                'epsilon_range': [float(self.epsilon.min()), float(self.epsilon.max())],
                'I_range': [float(self.I.min()), float(self.I.max())],
                'n_data_points': len(self.epsilon)
            },
            'conclusion': self._generate_conclusion()
        }
        
        return report
    
    def _generate_conclusion(self):
        """Generate text conclusion based on analysis"""
        if self.p_fit is None:
            self.fit_scaling_exponent()
        
        deviation = abs(self.p_fit - 2.0)
        
        if deviation < 0.001:
            strength = "EXCELLENT"
        elif deviation < 0.01:
            strength = "STRONG"
        elif deviation < 0.05:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        conclusion = f"""
SCALING LAW VERIFICATION RESULTS
=================================

Fitted scaling exponent: p = {self.p_fit:.6f}
Theoretical prediction: p = 2.000000
Deviation from theory: {deviation:.6f} ({deviation*100:.4f}%)

Evidence strength: {strength}

Universal kernel K₂ = {self.K2_mean:.6f} ± {self.K2_std:.6f}
Relative uncertainty: {(self.K2_std/self.K2_mean*100):.4f}%

Conclusion: The numerical data {'SUPPORTS' if deviation < 0.01 else 'DOES NOT SUPPORT'} 
the universal quadratic scaling law I_nonadd ∝ ε² in the perturbative regime.
"""
        return conclusion


def load_experimental_data(data_path=None):
    """
    Load experimental data from file or use example data
    
    Parameters:
    -----------
    data_path : str or Path, optional
        Path to CSV file with columns: epsilon, I_max
    
    Returns:
    --------
    epsilon : array
        Perturbation amplitudes
    I_max : array
        Maximum non-additive observable values
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        epsilon = df['epsilon'].values
        I_max = df['I_max'].values
    else:
        # Use example data from the audit
        epsilon = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        I_max = np.array([1.543259e-9, 1.388952e-8, 1.543240e-7, 
                          1.388820e-6, 1.542762e-5, 1.387539e-4, 1.538148e-3])
        print("Using example data from Audit V4.2-V4.3")
    
    return epsilon, I_max


def main():
    """Main execution function"""
    print("=" * 60)
    print("UNIVERSAL NONLINEAR SCALING LAW VERIFICATION")
    print("=" * 60)
    print()
    
    # Load data
    epsilon, I_max = load_experimental_data()
    
    # Create verification instance
    verifier = ScalingLawVerification(epsilon, I_max)
    
    # Fit scaling exponent
    print("Fitting scaling exponent...")
    p_fit, C_fit = verifier.fit_scaling_exponent()
    print(f"Fitted exponent: p = {p_fit:.6f}")
    print(f"Amplitude constant: C = {C_fit:.6f}")
    print()
    
    # Compute K₂ kernel
    print("Computing universal kernel K₂...")
    K2_vals, K2_mean, K2_std = verifier.compute_K2_kernel()
    print(f"K₂ = {K2_mean:.6f} ± {K2_std:.6f}")
    print(f"Relative uncertainty: {K2_std/K2_mean*100:.4f}%")
    print()
    
    # Verify quadratic scaling
    print("Verifying quadratic scaling...")
    is_quadratic, deviation = verifier.verify_quadratic_scaling()
    print(f"Is quadratic scaling (tolerance=1%): {is_quadratic}")
    print(f"Deviation from p=2: {deviation:.6f}")
    print()
    
    # Generate report
    print("Generating analysis report...")
    report = verifier.generate_report()
    print(report['conclusion'])
    
    # Save results
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON report
    report_path = output_dir / "scaling_law_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")
    
    # Save CSV data
    df_results = pd.DataFrame({
        'epsilon': epsilon,
        'I_max': I_max,
        'K2': K2_vals,
        'I_theoretical': C_fit * epsilon**2,
        'relative_error_percent': (I_max - C_fit * epsilon**2) / (C_fit * epsilon**2) * 100
    })
    csv_path = output_dir / "scaling_law_data.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    # Create plots
    print("Creating analysis plots...")
    plot_path = output_dir / "scaling_law_analysis.png"
    verifier.plot_scaling_analysis(save_path=plot_path)
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()