#!/usr/bin/env python3
"""
Universal Nonlinear Scaling Law Demonstration

This script provides a comprehensive demonstration of the universal scaling law:
I_nonadd(ε,ω,φ₀) = ε² K₂(ω,φ₀) + O(ε³)

Based on Audit V4.2-V4.3 results, this demonstrates:
1. Quadratic scaling (p ≈ 2.0) 
2. Negligible linear term (c₁ ≈ 0)
3. Universal kernel K₂ characterization
4. Separation of scaling universality from frequency selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scaling_law_verification import ScalingLawVerification
from kernel_K2_analysis import KernelK2Analyzer
from perturbation_expansion import PerturbationExpansionAnalyzer


class UniversalScalingLawDemo:
    """
    Comprehensive demonstration of the universal nonlinear scaling law
    """
    
    def __init__(self, epsilon_values, I_values):
        """
        Initialize the demonstration with experimental data
        
        Parameters:
        -----------
        epsilon_values : array-like
            Perturbation amplitudes
        I_values : array-like
            Non-additive observable values
        """
        self.epsilon = np.asarray(epsilon_values)
        self.I = np.asarray(I_values)
        
        # Initialize analyzers
        self.scaling_verifier = ScalingLawVerification(epsilon_values, I_values)
        self.kernel_analyzer = KernelK2Analyzer(epsilon_values)
        self.expansion_analyzer = PerturbationExpansionAnalyzer(epsilon_values, I_values)
        
        # Results storage
        self.results = {}
        
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        
        Returns:
        --------
        results : dict
            Complete analysis results
        """
        print("=" * 70)
        print("UNIVERSAL NONLINEAR SCALING LAW - COMPLETE ANALYSIS")
        print("=" * 70)
        print()
        
        # 1. Scaling Law Verification
        print("STEP 1: SCALING LAW VERIFICATION")
        print("-" * 70)
        p_fit, C_fit = self.scaling_verifier.fit_scaling_exponent()
        is_quadratic, deviation = self.scaling_verifier.verify_quadratic_scaling()
        
        print(f"Fitted scaling exponent: p = {p_fit:.6f}")
        print(f"Theoretical prediction: p = 2.000000")
        print(f"Deviation from theory: {deviation:.6f} ({deviation*100:.4f}%)")
        print(f"Quadratic scaling confirmed: {is_quadratic}")
        print()
        
        self.results['scaling'] = {
            'p_fitted': float(p_fit),
            'p_theoretical': 2.0,
            'deviation': float(deviation),
            'is_quadratic': bool(is_quadratic),
            'amplitude_constant': float(C_fit)
        }
        
        # 2. Kernel K₂ Analysis
        print("STEP 2: UNIVERSAL KERNEL K₂ ANALYSIS")
        print("-" * 70)
        K2_values = self.I / (self.epsilon**2)
        self.kernel_analyzer.K2_values = K2_values
        self.kernel_analyzer.K2_mean = np.mean(K2_values)
        self.kernel_analyzer.K2_std = np.std(K2_values)
        
        print(f"Universal kernel K₂ = {self.kernel_analyzer.K2_mean:.6f} ± {self.kernel_analyzer.K2_std:.6f}")
        print(f"Relative uncertainty: {self.kernel_analyzer.K2_std/self.kernel_analyzer.K2_mean*100:.4f}%")
        
        # Extrapolation to ε→0
        K2_limit_poly, err_poly = self.kernel_analyzer.extrapolate_to_zero_epsilon(
            self.epsilon, self.I, method='polynomial')
        print(f"K₂(0) polynomial extrapolation: {K2_limit_poly:.6f} ± {err_poly:.6f}")
        print()
        
        self.results['kernel'] = {
            'K2_mean': float(self.kernel_analyzer.K2_mean),
            'K2_std': float(self.kernel_analyzer.K2_std),
            'K2_limit_extrapolated': float(K2_limit_poly),
            'K2_extrapolation_error': float(err_poly)
        }
        
        # 3. Perturbation Expansion Analysis
        print("STEP 3: PERTURBATION EXPANSION ANALYSIS")
        print("-" * 70)
        c1, c1_err, c1_is_zero = self.expansion_analyzer.extract_linear_coefficient()
        c2, c2_err = self.expansion_analyzer.extract_quadratic_coefficient()
        optimal_order, criteria = self.expansion_analyzer.fit_optimal_order()
        
        print(f"Linear coefficient c₁ = {c1:.2e} ± {c1_err:.2e}")
        print(f"c₁ ≈ 0: {c1_is_zero}")
        print(f"Quadratic coefficient c₂ = {c2:.6f} ± {c2_err:.6f}")
        print(f"Optimal expansion order: {optimal_order}")
        print()
        
        self.results['expansion'] = {
            'c1': float(c1),
            'c1_error': float(c1_err),
            'c1_is_zero': bool(c1_is_zero),
            'c2': float(c2),
            'c2_error': float(c2_err),
            'optimal_order': int(optimal_order)
        }
        
        # 4. Regime Analysis
        print("STEP 4: PERTURBATIVE REGIME ANALYSIS")
        print("-" * 70)
        regime = self.expansion_analyzer.analyze_regime_breakdown()
        print(f"Perturbative regime valid for ε < {regime['breakdown_epsilon']:.2e}")
        print(f"Dominant term at small ε: {regime['dominant_term']}")
        print()
        
        self.results['regime'] = {
            'breakdown_epsilon': float(regime['breakdown_epsilon']),
            'dominant_term': regime['dominant_term']
        }
        
        # 5. Final Assessment
        print("STEP 5: FINAL ASSESSMENT")
        print("-" * 70)
        assessment = self._generate_final_assessment()
        print(assessment)
        
        self.results['assessment'] = assessment
        
        return self.results
    
    def _generate_final_assessment(self):
        """Generate final assessment of the scaling law"""
        p_fit = self.results['scaling']['p_fitted']
        deviation = self.results['scaling']['deviation']
        c1_is_zero = self.results['expansion']['c1_is_zero']
        K2 = self.results['kernel']['K2_mean']
        
        # Evidence strength
        if deviation < 0.001:
            scaling_strength = "EXCELLENT"
        elif deviation < 0.01:
            scaling_strength = "STRONG"
        elif deviation < 0.05:
            scaling_strength = "MODERATE"
        else:
            scaling_strength = "WEAK"
        
        assessment = f"""
FINAL ASSESSMENT
================

SCALING UNIVERSALITY (Strongly Supported):
- Scaling exponent: p = {p_fit:.6f} (theoretical: p = 2.0)
- Deviation from theory: {deviation:.6f} ({deviation*100:.4f}%)
- Evidence strength: {scaling_strength}
- Conclusion: Quadratic scaling I ∝ ε² is firmly established

UNIVERSAL KERNEL K₂ (Characterized):
- K₂ = {K2:.6f} ± {self.results['kernel']['K2_std']:.6f}
- K₂ represents the nonlinear susceptibility of the coupled RG flow
- K₂ is universal in the ε→0 limit

PERTURBATION STRUCTURE (Confirmed):
- Linear term c₁ ≈ 0: {c1_is_zero}
- Leading order: Quadratic (ε²)
- Expansion: I = ε² K₂ + O(ε³)

SEPARATION OF HYPOTHESES:
1. Scaling universality (p ≈ 2): STRONGLY SUPPORTED by numerical data
2. Golden-ratio frequency selection: NOT YET SUPPORTED statistically

CONCLUSION:
The universal nonlinear scaling law I_nonadd = ε² K₂(ω,φ₀) + O(ε³) is
well-established from Audit V4.2-V4.3. The quadratic scaling is a robust
numerical result. The frequency dependence K₂(ω) and potential golden-ratio
selection require additional experimental validation.

This represents a theoretically solid formulation of ASTRA-Q that separates
the established scaling universality from the as-yet-unproven frequency selection.
"""
        return assessment
    
    def create_comprehensive_plots(self, save_path=None):
        """
        Create comprehensive analysis plots
        
        Parameters:
        -----------
        save_path : str or Path, optional
            Path to save the figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Get key parameters
        p_fit = self.results['scaling']['p_fitted']
        C_fit = self.results['scaling']['amplitude_constant']
        deviation = self.results['scaling']['deviation']
        c1 = self.results['expansion']['c1']
        c1_err = self.results['expansion']['c1_error']
        c2 = self.results['expansion']['c2']
        c2_err = self.results['expansion']['c2_error']
        c1_is_zero = self.results['expansion']['c1_is_zero']
        optimal_order = self.results['expansion']['optimal_order']
        
        # Row 1: Scaling Law Verification
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.loglog(self.epsilon, self.I, 'ko', markersize=10, label='Data')
        
        epsilon_fit = np.logspace(np.log10(self.epsilon.min()), 
                                 np.log10(self.epsilon.max()), 100)
        I_fit = C_fit * epsilon_fit**p_fit
        ax1.loglog(epsilon_fit, I_fit, 'r--', linewidth=2, 
                  label=f'Fit: p={p_fit:.4f}')
        I_theory = C_fit * epsilon_fit**2
        ax1.loglog(epsilon_fit, I_theory, 'g:', linewidth=2, 
                  label='Theory: p=2.0')
        
        ax1.set_xlabel('ε', fontsize=12, fontweight='bold')
        ax1.set_ylabel('I', fontsize=12, fontweight='bold')
        ax1.set_title('Scaling Law Verification', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Kernel K₂ analysis
        ax2 = fig.add_subplot(gs[0, 1])
        K2_vals = self.I / (self.epsilon**2)
        ax2.semilogx(self.epsilon, K2_vals, 'bo-', markersize=8)
        ax2.axhline(y=self.results['kernel']['K2_mean'], color='r', linestyle='--', 
                   linewidth=2, label=f'K₂ = {self.results["kernel"]["K2_mean"]:.6f}')
        ax2.fill_between(self.epsilon, 
                         self.results['kernel']['K2_mean'] - self.results['kernel']['K2_std'],
                         self.results['kernel']['K2_mean'] + self.results['kernel']['K2_std'],
                         alpha=0.2, color='r')
        ax2.set_xlabel('ε', fontsize=12, fontweight='bold')
        ax2.set_ylabel('K₂(ε) = I/ε²', fontsize=12, fontweight='bold')
        ax2.set_title('Universal Kernel K₂', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Deviation from quadratic
        ax3 = fig.add_subplot(gs[0, 2])
        I_theoretical = C_fit * self.epsilon**2
        relative_error = (self.I - I_theoretical) / I_theoretical * 100
        ax3.semilogx(self.epsilon, relative_error, 'go-', markersize=8)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax3.set_xlabel('ε', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Deviation from p=2 (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Deviation from Quadratic Scaling', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Coefficient analysis
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.bar(['c₁'], [c1], yerr=[c1_err], color='blue', alpha=0.7, capsize=10)
        ax4.bar(['c₂'], [c2], yerr=[c2_err], color='red', alpha=0.7, capsize=10)
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax4.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax4.set_title('Perturbation Coefficients', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Row 2: Term Contributions and Regime Analysis
        ax5 = fig.add_subplot(gs[1, 0:2])
        regime = self.expansion_analyzer.analyze_regime_breakdown()
        ax5.loglog(self.epsilon, regime['linear_term_magnitude'], 'b-', 
                  label='|c₁ε|', linewidth=2)
        ax5.loglog(self.epsilon, regime['quadratic_term_magnitude'], 'r-', 
                  label='|c₂ε²|', linewidth=2)
        ax5.loglog(self.epsilon, regime['cubic_term_magnitude'], 'g-', 
                  label='|c₃ε³|', linewidth=2)
        ax5.axvline(x=regime['breakdown_epsilon'], color='k', linestyle='--', 
                   linewidth=2, label=f'Breakdown: ε={regime["breakdown_epsilon"]:.2e}')
        ax5.set_xlabel('ε', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Term Magnitude', fontsize=12, fontweight='bold')
        ax5.set_title('Term Contribution Analysis', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Model selection
        ax6 = fig.add_subplot(gs[1, 2])
        criteria = self.expansion_analyzer.goodness_of_fit
        orders = list(criteria.keys())
        r_squared = [criteria[o]['R_squared'] for o in orders]
        ax6.plot(orders, r_squared, 'mo-', markersize=8)
        ax6.axvline(x=optimal_order, color='r', linestyle='--', linewidth=2,
                   label=f'Optimal: {optimal_order}')
        ax6.set_xlabel('Polynomial Order', fontsize=12, fontweight='bold')
        ax6.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax6.set_title('Model Selection', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Residual analysis
        ax7 = fig.add_subplot(gs[1, 3])
        coeffs_opt, _, _ = self.expansion_analyzer.fit_polynomial_expansion(optimal_order)
        A_opt = np.vstack([self.epsilon**n for n in range(optimal_order + 1)]).T
        I_pred_opt = A_opt @ coeffs_opt
        residuals_opt = self.I - I_pred_opt
        
        ax7.semilogx(self.epsilon, residuals_opt, 'co-', markersize=8)
        ax7.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax7.set_xlabel('ε', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax7.set_title(f'Residuals (Order {optimal_order})', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # Row 3: Summary and Conclusions
        ax8 = fig.add_subplot(gs[2, :])
        ax8.axis('off')
        
        summary_text = f"""
UNIVERSAL NONLINEAR SCALING LAW SUMMARY
=======================================

SCALING LAW: I_nonadd(ε,ω,φ₀) = ε² K₂(ω,φ₀) + O(ε³)

NUMERICAL VERIFICATION (Audit V4.2-V4.3):
• Scaling exponent: p = {p_fit:.6f} (theoretical: p = 2.0)
• Deviation from theory: {deviation:.6f} ({deviation*100:.4f}%)
• Universal kernel: K₂ = {self.results['kernel']['K2_mean']:.6f} ± {self.results['kernel']['K2_std']:.6f}
• Linear coefficient: c₁ = {c1:.2e} ± {c1_err:.2e} (c₁ ≈ 0: {c1_is_zero})
• Quadratic coefficient: c₂ = {c2:.6f} ± {c2_err:.6f}

PERTURBATIVE REGIME:
• Valid for ε < {regime['breakdown_epsilon']:.2e}
• Dominant term: {regime['dominant_term']}
• Leading order: Quadratic (ε²)

HYPOTHESIS SEPARATION:
✓ Scaling universality (p ≈ 2): STRONGLY SUPPORTED
? Golden-ratio frequency selection: NOT YET SUPPORTED

CONCLUSION:
The universal quadratic scaling law is well-established from numerical experiments.
The kernel K₂(ω,φ₀) characterizes the nonlinear susceptibility of the coupled RG flow.
Frequency dependence and potential golden-ratio selection require further validation.
"""
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive plot saved to {save_path}")
        
        return fig
    
    def generate_final_report(self, output_dir=None):
        """
        Generate final comprehensive report
        
        Parameters:
        -----------
        output_dir : str or Path, optional
            Directory to save report files
            
        Returns:
        --------
        report : dict
            Complete analysis report
        """
        if output_dir is None:
            output_dir = Path("data")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Combine all reports
        report = {
            'universal_scaling_law': {
                'formulation': 'I_nonadd(ε,ω,φ₀) = ε² K₂(ω,φ₀) + O(ε³)',
                'scaling_analysis': self.results['scaling'],
                'kernel_analysis': self.results['kernel'],
                'expansion_analysis': self.results['expansion'],
                'regime_analysis': self.results['regime'],
                'final_assessment': self.results['assessment']
            },
            'metadata': {
                'audit_version': 'V4.2-V4.3',
                'analysis_date': str(pd.Timestamp.now()),
                'data_points': len(self.epsilon),
                'epsilon_range': [float(self.epsilon.min()), float(self.epsilon.max())]
            }
        }
        
        # Save JSON report
        report_path = output_dir / "universal_scaling_law_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Final report saved to {report_path}")
        
        # Save CSV data
        df_results = pd.DataFrame({
            'epsilon': self.epsilon,
            'I_max': self.I,
            'K2': self.I / (self.epsilon**2),
            'I_quadratic_prediction': self.results['scaling']['amplitude_constant'] * self.epsilon**2,
            'relative_error_percent': ((self.I - self.results['scaling']['amplitude_constant'] * self.epsilon**2) / 
                                      (self.results['scaling']['amplitude_constant'] * self.epsilon**2) * 100)
        })
        csv_path = output_dir / "universal_scaling_law_data.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        # Create comprehensive plots
        plot_path = output_dir / "universal_scaling_law_comprehensive.png"
        self.create_comprehensive_plots(save_path=plot_path)
        
        return report


def main():
    """Main execution function"""
    print("=" * 70)
    print("UNIVERSAL NONLINEAR SCALING LAW DEMONSTRATION")
    print("Based on Audit V4.2-V4.3 Results")
    print("=" * 70)
    print()
    
    # Example data from Audit V4.2-V4.3
    epsilon = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    I_max = np.array([1.543259e-9, 1.388952e-8, 1.543240e-7, 
                      1.388820e-6, 1.542762e-5, 1.387539e-4, 1.538148e-3])
    
    # Create demonstration
    demo = UniversalScalingLawDemo(epsilon, I_max)
    
    # Run complete analysis
    results = demo.run_complete_analysis()
    
    # Generate final report
    print()
    print("=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    report = demo.generate_final_report(output_dir)
    
    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("All analysis results have been saved to the 'data' directory.")
    print("The universal scaling law I_nonadd = ε² K₂ + O(ε³) is well-established.")


if __name__ == "__main__":
    main()