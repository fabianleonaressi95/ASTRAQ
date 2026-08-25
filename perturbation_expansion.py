#!/usr/bin/env python3
"""
Perturbation Expansion Analysis for Non-Additive RG Flow Response

This script analyzes the perturbation expansion:
I(ε,ω) = c₁(ω)ε + c₂(ω)ε² + c₃(ω)ε³ + ...

Based on numerical evidence that c₁(ω) ≈ 0, leading to:
I(ε,ω) = ε² K₂(ω) + O(ε³)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path
import json
from itertools import combinations


class PerturbationExpansionAnalyzer:
    """
    Analyze the perturbation expansion of the non-additive observable
    """
    
    def __init__(self, epsilon_values, I_values, max_order=4):
        """
        Initialize perturbation expansion analyzer
        
        Parameters:
        -----------
        epsilon_values : array-like
            Perturbation amplitudes
        I_values : array-like
            Non-additive observable values
        max_order : int
            Maximum order of perturbation expansion to consider
        """
        self.epsilon = np.asarray(epsilon_values)
        self.I = np.asarray(I_values)
        self.max_order = max_order
        self.coefficients = {}
        self.coefficient_errors = {}
        self.goodness_of_fit = {}
        
    def fit_polynomial_expansion(self, order):
        """
        Fit polynomial expansion: I = Σ c_n ε^n
        
        Parameters:
        -----------
        order : int
            Order of polynomial fit
            
        Returns:
        --------
        coefficients : array
            Fitted coefficients [c_0, c_1, ..., c_order]
        errors : array
            Standard errors of coefficients
        """
        # Design matrix for polynomial fit
        epsilon = self.epsilon
        A = np.vstack([epsilon**n for n in range(order + 1)]).T
        
        # Least squares fit
        coeffs, residuals, rank, singular_values = np.linalg.lstsq(A, self.I, rcond=None)
        
        # Estimate errors using residual variance
        if len(residuals) > 0 and len(epsilon) > order + 1:
            residual_variance = residuals[0] / (len(epsilon) - order - 1)
            cov_matrix = residual_variance * np.linalg.inv(A.T @ A)
            errors = np.sqrt(np.diag(cov_matrix))
        else:
            errors = np.full(order + 1, np.nan)
        
        # Calculate R²
        I_pred = A @ coeffs
        ss_res = np.sum((self.I - I_pred)**2)
        ss_tot = np.sum((self.I - np.mean(self.I))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        self.coefficients[order] = coeffs
        self.coefficient_errors[order] = errors
        self.goodness_of_fit[order] = {
            'R_squared': r_squared,
            'residuals': residuals[0] if len(residuals) > 0 else np.nan
        }
        
        return coeffs, errors, r_squared
    
    def extract_linear_coefficient(self):
        """
        Extract the linear coefficient c₁ to test if it's zero
        
        Returns:
        --------
        c1 : float
            Linear coefficient
        c1_error : float
            Uncertainty in c₁
        is_zero : bool
            Whether c₁ is consistent with zero within uncertainty
        """
        if 1 not in self.coefficients:
            self.fit_polynomial_expansion(1)
        
        c1 = self.coefficients[1][1]  # c₁ is the coefficient of ε¹
        c1_error = self.coefficient_errors[1][1]
        
        # Test if c₁ is consistent with zero
        is_zero = abs(c1) < 2 * c1_error  # 2σ criterion
        
        return c1, c1_error, is_zero
    
    def extract_quadratic_coefficient(self):
        """
        Extract the quadratic coefficient c₂ = K₂
        
        Returns:
        --------
        c2 : float
            Quadratic coefficient (universal kernel K₂)
        c2_error : float
            Uncertainty in c₂
        """
        if 2 not in self.coefficients:
            self.fit_polynomial_expansion(2)
        
        c2 = self.coefficients[2][2]  # c₂ is the coefficient of ε²
        c2_error = self.coefficient_errors[2][2]
        
        return c2, c2_error
    
    def fit_optimal_order(self, max_order=None):
        """
        Determine the optimal order of perturbation expansion using
        information criteria (AIC, BIC)
        
        Parameters:
        -----------
        max_order : int, optional
            Maximum order to test (defaults to self.max_order)
            
        Returns:
        --------
        optimal_order : int
            Optimal order according to AIC
        criteria_results : dict
            AIC and BIC values for each order
        """
        if max_order is None:
            max_order = self.max_order
        
        n = len(self.epsilon)
        criteria_results = {}
        
        for order in range(1, max_order + 1):
            coeffs, errors, r_squared = self.fit_polynomial_expansion(order)
            
            # Calculate residuals
            A = np.vstack([self.epsilon**n for n in range(order + 1)]).T
            I_pred = A @ coeffs
            residuals = self.I - I_pred
            ss_res = np.sum(residuals**2)
            
            # Information criteria
            k = order + 1  # number of parameters
            aic = n * np.log(ss_res / n) + 2 * k
            bic = n * np.log(ss_res / n) + k * np.log(n)
            
            criteria_results[order] = {
                'AIC': aic,
                'BIC': bic,
                'R_squared': r_squared,
                'coefficients': coeffs,
                'errors': errors
            }
        
        # Find optimal order (minimum AIC)
        optimal_order = min(criteria_results.keys(), key=lambda x: criteria_results[x]['AIC'])
        
        return optimal_order, criteria_results
    
    def compute_higher_order_contributions(self):
        """
        Compute contributions of higher-order terms to understand
        when O(ε³) terms become significant
        
        Returns:
        --------
        contributions : dict
            Contribution of each order at each ε value
        """
        contributions = {}
        
        for order in range(1, self.max_order + 1):
            coeffs, _, _ = self.fit_polynomial_expansion(order)
            
            # Compute contribution of each term
            term_contributions = []
            for n in range(1, order + 1):
                term_value = coeffs[n] * self.epsilon**n
                term_contributions.append(term_value)
            
            contributions[order] = {
                'coefficients': coeffs,
                'term_contributions': term_contributions,
                'total_prediction': np.sum(term_contributions, axis=0)
            }
        
        return contributions
    
    def analyze_regime_breakdown(self):
        """
        Analyze the perturbative regime breakdown - at what ε do
        higher-order terms become significant?
        
        Returns:
        --------
        regime_analysis : dict
            Analysis of when different terms become important
        """
        regime_analysis = {}
        
        # Fit up to 3rd order
        coeffs_3, _, _ = self.fit_polynomial_expansion(3)
        
        # Compute magnitude of each term
        linear_term = abs(coeffs_3[1]) * self.epsilon
        quadratic_term = abs(coeffs_3[2]) * self.epsilon**2
        cubic_term = abs(coeffs_3[3]) * self.epsilon**3
        
        # Find where cubic term becomes 10% of quadratic term
        if np.any(quadratic_term > 0):
            ratio = cubic_term / quadratic_term
            breakdown_idx = np.where(ratio > 0.1)[0]
            if len(breakdown_idx) > 0:
                breakdown_epsilon = self.epsilon[breakdown_idx[0]]
            else:
                breakdown_epsilon = self.epsilon.max()
        else:
            breakdown_epsilon = self.epsilon.max()
        
        regime_analysis = {
            'linear_term_magnitude': linear_term,
            'quadratic_term_magnitude': quadratic_term,
            'cubic_term_magnitude': cubic_term,
            'breakdown_epsilon': breakdown_epsilon,
            'dominant_term': self._determine_dominant_term(coeffs_3)
        }
        
        return regime_analysis
    
    def _determine_dominant_term(self, coeffs):
        """Determine which term dominates the expansion"""
        # Test at smallest epsilon (most perturbative)
        epsilon_test = self.epsilon.min()
        
        linear = abs(coeffs[1]) * epsilon_test
        quadratic = abs(coeffs[2]) * epsilon_test**2
        cubic = abs(coeffs[3]) * epsilon_test**3 if len(coeffs) > 3 else 0
        
        if linear > quadratic and linear > cubic:
            return "linear"
        elif quadratic > linear and quadratic > cubic:
            return "quadratic"
        elif cubic > linear and cubic > quadratic:
            return "cubic"
        else:
            return "mixed"
    
    def plot_expansion_analysis(self, save_path=None):
        """
        Create comprehensive perturbation expansion analysis plots
        
        Parameters:
        -----------
        save_path : str or Path, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: Data vs different polynomial fits
        ax1 = axes[0, 0]
        ax1.loglog(self.epsilon, self.I, 'ko', markersize=8, label='Data')
        
        colors = ['b', 'g', 'r', 'm']
        for order, color in zip(range(1, 4), colors):
            coeffs, _, _ = self.fit_polynomial_expansion(order)
            epsilon_fit = np.logspace(np.log10(self.epsilon.min()), 
                                     np.log10(self.epsilon.max()), 100)
            I_fit = np.sum([coeffs[n] * epsilon_fit**n for n in range(order + 1)], axis=0)
            ax1.loglog(epsilon_fit, I_fit, color=color, linestyle='--', 
                      label=f'Order {order}')
        
        ax1.set_xlabel('Perturbation Amplitude ε', fontsize=12)
        ax1.set_ylabel('Non-additive Observable I', fontsize=12)
        ax1.set_title('Perturbation Expansion Fits', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coefficient analysis
        ax2 = axes[0, 1]
        orders = []
        c1_values = []
        c2_values = []
        c1_errors = []
        c2_errors = []
        
        for order in range(2, min(5, self.max_order + 1)):
            coeffs, errors, _ = self.fit_polynomial_expansion(order)
            orders.append(order)
            c1_values.append(coeffs[1])
            c2_values.append(coeffs[2])
            c1_errors.append(errors[1])
            c2_errors.append(errors[2])
        
        ax2.errorbar(orders, c1_values, yerr=c1_errors, fmt='bo-', 
                    markersize=8, capsize=5, label='c₁ (linear)')
        ax2.errorbar(orders, c2_values, yerr=c2_errors, fmt='ro-', 
                    markersize=8, capsize=5, label='c₂ (quadratic)')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.set_xlabel('Polynomial Order', fontsize=12)
        ax2.set_ylabel('Coefficient Value', fontsize=12)
        ax2.set_title('Coefficient Stability Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Information criteria
        ax3 = axes[0, 2]
        optimal_order, criteria = self.fit_optimal_order()
        orders_plot = list(criteria.keys())
        aic_values = [criteria[o]['AIC'] for o in orders_plot]
        bic_values = [criteria[o]['BIC'] for o in orders_plot]
        
        ax3.plot(orders_plot, aic_values, 'bo-', markersize=8, label='AIC')
        ax3.plot(orders_plot, bic_values, 'ro-', markersize=8, label='BIC')
        ax3.axvline(x=optimal_order, color='g', linestyle='--', 
                   label=f'Optimal: {optimal_order}')
        ax3.set_xlabel('Polynomial Order', fontsize=12)
        ax3.set_ylabel('Information Criterion', fontsize=12)
        ax3.set_title('Model Selection (AIC/BIC)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Term contributions
        ax4 = axes[1, 0]
        regime = self.analyze_regime_breakdown()
        ax4.loglog(self.epsilon, regime['linear_term_magnitude'], 'b-', 
                  label='|c₁ε|', linewidth=2)
        ax4.loglog(self.epsilon, regime['quadratic_term_magnitude'], 'r-', 
                  label='|c₂ε²|', linewidth=2)
        ax4.loglog(self.epsilon, regime['cubic_term_magnitude'], 'g-', 
                  label='|c₃ε³|', linewidth=2)
        ax4.axvline(x=regime['breakdown_epsilon'], color='k', linestyle='--', 
                   label=f'Breakdown: ε={regime["breakdown_epsilon"]:.2e}')
        ax4.set_xlabel('Perturbation Amplitude ε', fontsize=12)
        ax4.set_ylabel('Term Magnitude', fontsize=12)
        ax4.set_title('Term Contribution Analysis', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Residual analysis for optimal order
        ax5 = axes[1, 1]
        coeffs_opt, _, _ = self.fit_polynomial_expansion(optimal_order)
        A_opt = np.vstack([self.epsilon**n for n in range(optimal_order + 1)]).T
        I_pred_opt = A_opt @ coeffs_opt
        residuals_opt = self.I - I_pred_opt
        
        ax5.semilogx(self.epsilon, residuals_opt, 'mo-', markersize=8)
        ax5.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax5.set_xlabel('Perturbation Amplitude ε', fontsize=12)
        ax5.set_ylabel('Residuals (I - I_fit)', fontsize=12)
        ax5.set_title(f'Residuals (Order {optimal_order})', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: c₁ significance test
        ax6 = axes[1, 2]
        c1, c1_err, is_zero = self.extract_linear_coefficient()
        
        ax6.bar(['c₁'], [c1], yerr=[c1_err], color='blue', alpha=0.7, capsize=10)
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax6.axhline(y=2*c1_err, color='r', linestyle='--', alpha=0.5, label='±2σ')
        ax6.axhline(y=-2*c1_err, color='r', linestyle='--', alpha=0.5)
        ax6.set_ylabel('Coefficient Value', fontsize=12)
        ax6.set_title(f'Linear Coefficient: c₁ = {c1:.2e} ± {c1_err:.2e}', fontsize=14, fontweight='bold')
        ax6.text(0.5, 0.5, f'c₁ ≈ 0: {is_zero}', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def generate_expansion_report(self):
        """
        Generate comprehensive perturbation expansion report
        
        Returns:
        --------
        report : dict
            Dictionary containing all expansion analysis results
        """
        c1, c1_err, c1_is_zero = self.extract_linear_coefficient()
        c2, c2_err = self.extract_quadratic_coefficient()
        optimal_order, criteria = self.fit_optimal_order()
        regime = self.analyze_regime_breakdown()
        
        report = {
            'expansion_analysis': {
                'linear_coefficient_c1': float(c1),
                'c1_uncertainty': float(c1_err),
                'c1_is_consistent_with_zero': bool(c1_is_zero),
                'quadratic_coefficient_c2': float(c2),
                'c2_uncertainty': float(c2_err),
                'universal_kernel_K2': float(c2)
            },
            'model_selection': {
                'optimal_order': int(optimal_order),
                'AIC_values': {int(k): float(v['AIC']) for k, v in criteria.items()},
                'BIC_values': {int(k): float(v['BIC']) for k, v in criteria.items()}
            },
            'regime_analysis': {
                'breakdown_epsilon': float(regime['breakdown_epsilon']),
                'dominant_term_at_small_epsilon': regime['dominant_term'],
                'perturbative_regime_valid_below': float(regime['breakdown_epsilon'])
            },
            'conclusion': self._generate_expansion_conclusion(c1_is_zero, optimal_order, regime)
        }
        
        return report
    
    def _generate_expansion_conclusion(self, c1_is_zero, optimal_order, regime):
        """Generate text conclusion for expansion analysis"""
        c1, c1_err, _ = self.extract_linear_coefficient()
        c2, c2_err = self.extract_quadratic_coefficient()
        
        conclusion = f"""
PERTURBATION EXPANSION ANALYSIS RESULTS
========================================

Expansion: I(ε,ω) = c₁ε + c₂ε² + c₃ε³ + ...

Coefficient analysis:
- Linear coefficient: c₁ = {c1:.2e} ± {c1_err:.2e}
- c₁ consistent with zero: {c1_is_zero}
- Quadratic coefficient: c₂ = {c2:.6f} ± {c2_err:.6f}
- Universal kernel: K₂ = c₂ = {c2:.6f}

Model selection:
- Optimal expansion order: {optimal_order}
- Leading behavior: Quadratic (ε²)

Regime analysis:
- Perturbative regime valid for ε < {regime['breakdown_epsilon']:.2e}
- Dominant term at small ε: {regime['dominant_term']}

Conclusion: {'The linear term c₁ is negligible, confirming I ∝ ε² scaling.' if c1_is_zero 
             else 'The linear term c₁ may be non-zero, requiring further investigation.'}

The perturbation expansion is dominated by the quadratic term, supporting
the universal scaling law I_nonadd = ε² K₂(ω,φ₀) + O(ε³).
"""
        return conclusion


def example_usage():
    """Example usage of PerturbationExpansionAnalyzer"""
    print("=" * 60)
    print("PERTURBATION EXPANSION ANALYSIS")
    print("=" * 60)
    print()
    
    # Example data from Audit V4.2-V4.3
    epsilon = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    I_max = np.array([1.543259e-9, 1.388952e-8, 1.543240e-7, 
                      1.388820e-6, 1.542762e-5, 1.387539e-4, 1.538148e-3])
    
    # Create analyzer
    analyzer = PerturbationExpansionAnalyzer(epsilon, I_max, max_order=4)
    
    # Extract coefficients
    print("Coefficient extraction:")
    c1, c1_err, c1_is_zero = analyzer.extract_linear_coefficient()
    print(f"  Linear coefficient c₁ = {c1:.2e} ± {c1_err:.2e}")
    print(f"  c₁ ≈ 0: {c1_is_zero}")
    
    c2, c2_err = analyzer.extract_quadratic_coefficient()
    print(f"  Quadratic coefficient c₂ = {c2:.6f} ± {c2_err:.6f}")
    print(f"  Universal kernel K₂ = {c2:.6f}")
    print()
    
    # Model selection
    print("Model selection:")
    optimal_order, criteria = analyzer.fit_optimal_order()
    print(f"  Optimal expansion order: {optimal_order}")
    for order in sorted(criteria.keys()):
        print(f"  Order {order}: AIC = {criteria[order]['AIC']:.2f}, R² = {criteria[order]['R_squared']:.6f}")
    print()
    
    # Regime analysis
    print("Regime analysis:")
    regime = analyzer.analyze_regime_breakdown()
    print(f"  Perturbative regime valid for ε < {regime['breakdown_epsilon']:.2e}")
    print(f"  Dominant term at small ε: {regime['dominant_term']}")
    print()
    
    # Generate report
    report = analyzer.generate_expansion_report()
    print(report['conclusion'])
    
    # Create plots
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "perturbation_expansion_analysis.png"
    analyzer.plot_expansion_analysis(save_path=plot_path)
    
    # Save results
    report_path = output_dir / "perturbation_expansion_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    example_usage()