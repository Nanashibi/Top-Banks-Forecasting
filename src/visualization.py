#!/usr/bin/env python3
"""
Banking Analytics Visualization Module
Creates comprehensive interactive charts and dashboards for banking forecasting results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime

class BankingVisualizer:
    def __init__(self, forecast_results=None, pivot_data=None, cleaned_df=None, forecast_df=None, forecast_quarters=None):
        """Initialize the Banking Visualizer with forecasting results"""
        self.forecast_results = forecast_results
        self.pivot_data = pivot_data
        self.cleaned_df = cleaned_df
        self.forecast_df = forecast_df
        self.forecast_quarters = forecast_quarters or [
            '2025-Q3', '2025-Q4', '2026-Q1', 
            '2026-Q2', '2026-Q3', '2026-Q4'
        ]
        
        print("📊 Banking Visualization System Initialized")
    
    def create_all_visualizations(self):
        """Create comprehensive visualization dashboard"""
        print("\n📊 Creating complete visualization dashboard...")
        
        # Create output directory
        charts_dir = "results/charts"
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. Historical trends overview
        self.create_historical_trends_chart(charts_dir)
        
        # 2. Bank comparison charts
        self.create_bank_comparison_charts(charts_dir)
        
        # 3. Forecast visualization
        self.create_forecast_charts(charts_dir)
        
        # 4. Performance metrics dashboard
        self.create_performance_dashboard(charts_dir)
        
        # 5. Advanced analytics charts
        self.create_growth_analysis_chart(charts_dir)
        self.create_risk_assessment_chart(charts_dir)
        
        print(f"✅ All visualizations saved to: {charts_dir}/")
        return charts_dir
    
    def create_historical_trends_chart(self, output_dir):
        """Create historical trends overview chart"""
        if not self.pivot_data:
            print("⚠️ No pivot data available for historical trends")
            return
            
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Net Income Trends', 'Total Assets Trends', 'Deposits Trends'],
            vertical_spacing=0.08
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, concept in enumerate(['NetIncome', 'TotalAssets', 'Deposits'], 1):
            data = self.pivot_data[concept]
            
            for j, bank in enumerate(data.columns[:5]):  # Top 5 banks
                series = data[bank].dropna()
                
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode='lines+markers',
                        name=f"{bank}" if i == 1 else None,
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=(i == 1),
                        legendgroup=f"bank_{j}"
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title="Banking Industry Historical Trends (Q1 2022 - Q2 2025)",
            height=1000,
            template="plotly_white"
        )
        
        fig.write_html(f"{output_dir}/historical_trends.html")
        print("  ✅ Historical trends chart created")
        return fig
    
    def create_forecast_charts(self, output_dir):
        """Create forecast visualization charts"""
        if not self.forecast_results:
            print("⚠️ No forecast results available")
            return
        
        for concept in ['NetIncome', 'TotalAssets', 'Deposits']:
            fig = go.Figure()
            
            concept_forecasts = self.forecast_results.get(concept, {})
            colors = px.colors.qualitative.Set1
            
            for i, (bank, results) in enumerate(concept_forecasts.items()):
                if i >= 5:  # Limit to top 5 banks for clarity
                    break
                
                historical = results['historical_data']
                forecast = results['ensemble_forecast']
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical.index,
                    y=historical.values,
                    mode='lines+markers',
                    name=f"{bank} (Historical)",
                    line=dict(color=colors[i % len(colors)]),
                    legendgroup=f"bank_{i}"
                ))
                
                # Forecast data
                fig.add_trace(go.Scatter(
                    x=self.forecast_quarters,
                    y=forecast,
                    mode='lines+markers',
                    name=f"{bank} (Forecast)",
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    legendgroup=f"bank_{i}"
                ))
            
            fig.update_layout(
                title=f"{concept} Forecasting: Historical vs Predicted (2022-2026)",
                xaxis_title="Quarter",
                yaxis_title=f"{concept} (USD)",
                template="plotly_white",
                height=600
            )
            
            fig.write_html(f"{output_dir}/{concept.lower()}_forecast.html")
        
        print("  ✅ Forecast charts created")
    
    def create_bank_comparison_charts(self, output_dir):
        """Create bank performance comparison charts"""
        if not hasattr(self, 'cleaned_df') or self.cleaned_df is None:
            print("⚠️ No cleaned data available for bank comparison")
            return
            
        # Latest quarter performance comparison
        latest_quarter = self.cleaned_df['quarter'].max()
        latest_data = self.cleaned_df[self.cleaned_df['quarter'] == latest_quarter]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Net Income', 'Total Assets', 'Deposits'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, concept in enumerate(['NetIncome', 'TotalAssets', 'Deposits'], 1):
            concept_data = latest_data[latest_data['concept'] == concept].nlargest(5, 'value')
            
            fig.add_trace(
                go.Bar(
                    x=concept_data['name'],
                    y=concept_data['value'],
                    name=concept,
                    showlegend=(i == 1)
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            title=f"Top 5 Banks Performance Comparison - {latest_quarter}",
            height=500,
            template="plotly_white"
        )
        
        fig.write_html(f"{output_dir}/bank_comparison.html")
        print("  ✅ Bank comparison chart created")
        return fig
    
    def create_performance_dashboard(self, output_dir):
        """Create executive performance dashboard"""
        if not self.forecast_results:
            print("⚠️ No forecast results available for performance dashboard")
            return
        
        # Calculate growth rates and performance metrics
        dashboard_data = []
        
        for concept in ['NetIncome', 'TotalAssets', 'Deposits']:
            concept_forecasts = self.forecast_results.get(concept, {})
            
            for bank, results in concept_forecasts.items():
                latest_actual = results['historical_data'].iloc[-1]
                forecast_2026_q4 = results['ensemble_forecast'][-1]
                
                # Calculate compound growth rate
                quarters_ahead = 6
                cagr = ((forecast_2026_q4 / latest_actual) ** (4/quarters_ahead) - 1) * 100
                
                dashboard_data.append({
                    'bank': bank,
                    'concept': concept,
                    'current_value': latest_actual,
                    'forecast_2026_q4': forecast_2026_q4,
                    'cagr': cagr
                })
        
        dashboard_df = pd.DataFrame(dashboard_data)
        
        # Save dashboard data
        os.makedirs(f"{output_dir}/../reports", exist_ok=True)
        dashboard_df.to_csv(f"{output_dir}/../reports/executive_dashboard.csv", index=False)
        print("  ✅ Executive dashboard data created")
        return dashboard_df
    
    def create_growth_analysis_chart(self, output_dir):
        """Create growth analysis visualization"""
        if not self.forecast_results:
            return
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Net Income Growth Rates', 'Total Assets Growth', 
                           'Deposits Growth', 'CAGR Comparison'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Pastel
        
        # Extract growth data for each concept
        for i, concept in enumerate(['NetIncome', 'TotalAssets', 'Deposits'], 1):
            concept_forecasts = self.forecast_results.get(concept, {})
            
            banks = []
            growth_rates = []
            
            for bank, results in concept_forecasts.items():
                if len(banks) >= 5:  # Limit to top 5 banks
                    break
                    
                latest_actual = results['historical_data'].iloc[-1]
                forecast_avg = results['ensemble_forecast'].mean()
                growth_rate = ((forecast_avg / latest_actual) - 1) * 100
                
                banks.append(bank)
                growth_rates.append(growth_rate)
            
            if i <= 3:
                row = 1 if i <= 2 else 2
                col = i if i <= 2 else i - 2
                
                fig.add_trace(
                    go.Bar(
                        x=banks,
                        y=growth_rates,
                        name=concept,
                        marker_color=colors[i-1]
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Banking Growth Analysis Dashboard",
            height=800,
            template="plotly_white",
            showlegend=False
        )
        
        fig.write_html(f"{output_dir}/growth_analysis.html")
        print("  ✅ Growth analysis chart created")
        return fig
    
    def create_risk_assessment_chart(self, output_dir):
        """Create risk assessment visualization"""
        if not self.forecast_results:
            return
            
        # Calculate volatility and confidence metrics
        risk_data = []
        
        for concept in ['NetIncome', 'TotalAssets', 'Deposits']:
            concept_forecasts = self.forecast_results.get(concept, {})
            
            for bank, results in concept_forecasts.items():
                historical = results['historical_data']
                volatility = historical.std() / historical.mean() * 100  # Coefficient of variation
                
                # Calculate forecast variance
                prophet_forecast = results.get('prophet_forecast', [])
                sarima_forecast = results.get('sarima_forecast', [])
                
                if prophet_forecast is not None and sarima_forecast is not None:
                    forecast_variance = np.var(prophet_forecast - sarima_forecast)
                    confidence_score = 100 - min(forecast_variance / 1e18, 50)  # Normalize
                else:
                    confidence_score = 50  # Medium confidence
                
                risk_data.append({
                    'bank': bank,
                    'concept': concept,
                    'volatility': volatility,
                    'confidence': confidence_score,
                    'risk_score': volatility * (100 - confidence_score) / 100
                })
        
        risk_df = pd.DataFrame(risk_data)
        
        # Create risk matrix
        fig = px.scatter(
            risk_df[risk_df['concept'] == 'NetIncome'],
            x='volatility',
            y='confidence',
            size='risk_score',
            color='bank',
            title='Banking Risk Assessment Matrix (Net Income)',
            labels={
                'volatility': 'Historical Volatility (%)',
                'confidence': 'Forecast Confidence (%)',
                'risk_score': 'Risk Score'
            }
        )
        
        fig.update_layout(
            template="plotly_white",
            height=600
        )
        
        fig.write_html(f"{output_dir}/risk_assessment.html")
        print("  ✅ Risk assessment chart created")
        return fig
    
    def create_interactive_dashboard(self):
        """Create a comprehensive interactive dashboard"""
        if not self.forecast_results:
            return None
            
        # Create a multi-panel dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Historical Trends', 'Forecast Comparison', 
                           'Growth Rates', 'Risk Matrix'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Add historical trends (Net Income only for space)
        if self.pivot_data and 'NetIncome' in self.pivot_data:
            data = self.pivot_data['NetIncome']
            colors = px.colors.qualitative.Set1
            
            for j, bank in enumerate(data.columns[:3]):  # Top 3 banks
                series = data[bank].dropna()
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode='lines+markers',
                        name=f"{bank}",
                        line=dict(color=colors[j % len(colors)])
                    ),
                    row=1, col=1
                )
        
        fig.update_layout(
            title="Banking Analytics Interactive Dashboard",
            height=800,
            template="plotly_white"
        )
        
        return fig

def load_and_visualize(forecast_results_path=None, data_path=None):
    """Convenience function to load data and create visualizations"""
    
    # Load forecast results if path provided
    forecast_results = None
    if forecast_results_path and os.path.exists(forecast_results_path):
        # Load from pickle or CSV depending on format
        pass
    
    # Create visualizer
    visualizer = BankingVisualizer(
        forecast_results=forecast_results
    )
    
    # Create all visualizations
    return visualizer.create_all_visualizations()

if __name__ == "__main__":
    # Example usage
    visualizer = BankingVisualizer()
    print("📊 Banking Visualization Module Ready")
    print("Use visualizer.create_all_visualizations() to generate charts")
