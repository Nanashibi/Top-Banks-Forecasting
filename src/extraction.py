#!/usr/bin/env python3
"""
SEC EDGAR Quarterly Data Extractor for Banking Analytics
Extracts quarterly financial data for top 100 banks from Q1 2022 to Q2 2025.
"""

import requests
import json
import csv
import time
import os
from datetime import datetime, timedelta
import pandas as pd

class SECEdgarExtractor:
    def __init__(self):
        self.base_url = "https://data.sec.gov/api/xbrl/companyconcept"
        self.headers = {
            'User-Agent': 'Banking Analytics Project (your-email@example.com)',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        # Essential forecasting metrics only (3 core banking indicators)
        self.financial_concepts = {
            'NetIncome': 'NetIncomeLoss',        # Profitability trend
            'TotalAssets': 'Assets',             # Growth trend  
            'Deposits': 'Deposits'               # Core banking metric
        }
        
        # Target quarters: Q1 2022 to Q2 2025
        self.target_quarters = self._generate_target_quarters()
        
    def _generate_target_quarters(self):
        """Generate list of target quarters from Q1 2022 to Q2 2025"""
        quarters = []
        
        # 2022: Q1, Q2, Q3, Q4
        for q in [1, 2, 3, 4]:
            quarters.append(f"2022-Q{q}")
            
        # 2023: Q1, Q2, Q3, Q4  
        for q in [1, 2, 3, 4]:
            quarters.append(f"2023-Q{q}")
            
        # 2024: Q1, Q2, Q3, Q4
        for q in [1, 2, 3, 4]:
            quarters.append(f"2024-Q{q}")
            
        # 2025: Q1, Q2
        for q in [1, 2]:
            quarters.append(f"2025-Q{q}")
            
        return quarters
    
    def _quarter_to_end_date(self, quarter_str):
        """Convert quarter string (e.g., '2024-Q1') to end date"""
        year, q = quarter_str.split('-Q')
        year = int(year)
        quarter = int(q)
        
        quarter_end_months = {1: 3, 2: 6, 3: 9, 4: 12}
        month = quarter_end_months[quarter]
        
        # Get last day of the quarter
        if month == 3:
            day = 31
        elif month == 6:
            day = 30
        elif month == 9:
            day = 30
        else:  # December
            day = 31
            
        return f"{year}-{month:02d}-{day:02d}"
    
    def extract_company_data(self, cik, ticker, company_name):
        """Extract quarterly data for a single company"""
        print(f"\n🏦 Extracting data for {company_name} ({ticker}) - CIK: {cik}")
        
        company_data = []
        successful_extractions = 0
        
        for concept_name, concept_code in self.financial_concepts.items():
            try:
                # Make API request
                url = f"{self.base_url}/CIK{cik:010d}/us-gaap/{concept_code}.json"
                response = requests.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract quarterly data
                    quarterly_data = self._extract_quarterly_values(data, concept_name)
                    company_data.extend(quarterly_data)
                    successful_extractions += 1
                    
                elif response.status_code == 404:
                    print(f"   ⚠️  Concept {concept_name} not found for {ticker}")
                    
                else:
                    print(f"   ❌ Error {response.status_code} for {concept_name}: {response.text[:100]}")
                
                # Rate limiting - SEC allows 10 requests per second
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ Exception extracting {concept_name}: {str(e)}")
                
        print(f"   ✅ Successfully extracted {successful_extractions}/{len(self.financial_concepts)} concepts")
        return company_data
    
    def _extract_quarterly_values(self, sec_data, concept_name):
        """Extract quarterly values from SEC API response"""
        quarterly_records = []
        
        try:
            # Get the units (usually USD)
            units = sec_data.get('units', {})
            
            # Try different unit types
            unit_data = None
            for unit_type in ['USD', 'usd', 'shares', 'pure']:
                if unit_type in units:
                    unit_data = units[unit_type]
                    break
                    
            if not unit_data:
                return quarterly_records
                
            # Process each filing
            for filing in unit_data:
                end_date = filing.get('end')
                form_type = filing.get('form', '')
                value = filing.get('val')
                
                if not end_date or value is None:
                    continue
                    
                # Check if this matches our target quarters
                quarter = self._date_to_quarter(end_date)
                if quarter in self.target_quarters:
                    
                    # Filter for quarterly reports (10-Q) and annual reports (10-K)
                    if form_type in ['10-Q', '10-K']:
                        record = {
                            'cik': sec_data.get('cik'),
                            'ticker': sec_data.get('entityName', '').split()[0] if sec_data.get('entityName') else '',
                            'concept': concept_name,
                            'quarter': quarter,
                            'end_date': end_date,
                            'value': value,
                            'form_type': form_type,
                            'unit': filing.get('unit', 'USD')
                        }
                        quarterly_records.append(record)
                        
        except Exception as e:
            print(f"     ⚠️  Error processing quarterly data for {concept_name}: {str(e)}")
            
        return quarterly_records
    
    def _date_to_quarter(self, date_str):
        """Convert date string to quarter format (e.g., '2024-Q1')"""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year = date_obj.year
            month = date_obj.month
            
            # Determine quarter based on month
            if month <= 3:
                quarter = 1
            elif month <= 6:
                quarter = 2
            elif month <= 9:
                quarter = 3
            else:
                quarter = 4
                
            return f"{year}-Q{quarter}"
            
        except:
            return None
    
    def extract_all_banks(self, banks_csv_path):
        """Extract data for top 10 US banks"""
        print("🚀 Starting SEC EDGAR data extraction for top 10 US banks...")
        print(f"📅 Target period: Q1 2022 to Q2 2025 ({len(self.target_quarters)} quarters)")
        print(f"📊 Extracting {len(self.financial_concepts)} essential forecasting metrics per bank")
        
        # Read top 10 US banks data
        banks_data = []
        with open(banks_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            banks_data = list(reader)
            
        print(f"🏦 Processing {len(banks_data)} top US banks for forecasting")
        
        # Extract data for each bank
        all_data = []
        successful_banks = 0
        
        for i, bank in enumerate(banks_data, 1):
            try:
                cik = int(bank['cik'])
                ticker = bank['ticker']
                name = bank['name']
                
                print(f"\n[{i}/{len(banks_data)}] Processing {name} ({ticker})...")
                
                # Extract company data
                company_data = self.extract_company_data(cik, ticker, name)
                
                # Add bank metadata to each record
                for record in company_data:
                    record.update({
                        'rank': bank['rank'],
                        'name': name,
                        'marketcap': bank['marketcap'],
                        'country': bank['country'],
                        'exchange': bank['exchange']
                    })
                    
                all_data.extend(company_data)
                successful_banks += 1
                
            except Exception as e:
                print(f"❌ Error processing {bank.get('name', 'Unknown')}: {str(e)}")
                
        print(f"\n📈 Final Summary: {successful_banks}/{len(banks_data)} banks processed successfully")
        print(f"📊 Total forecasting data points extracted: {len(all_data)}")
        
        return all_data, successful_banks
    
    def save_data(self, data, output_path):
        """Save extracted data to CSV"""
        if not data:
            print("❌ No data to save!")
            return
            
        print(f"\n💾 Saving {len(data)} records to {output_path}")
        
        # Define column order
        columns = [
            'rank', 'name', 'ticker', 'cik', 'country', 'exchange', 'marketcap',
            'quarter', 'end_date', 'concept', 'value', 'form_type', 'unit'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
            
        print(f"✅ Data saved successfully!")
        
        # Generate summary statistics
        self._generate_summary(data)
    
    def _generate_summary(self, data):
        """Generate and display summary statistics"""
        print(f"\n📊 EXTRACTION SUMMARY")
        print(f"{'='*50}")
        
        # Total records
        print(f"Total records extracted: {len(data):,}")
        
        # Unique banks
        unique_banks = len(set(record['name'] for record in data))
        print(f"Banks with data: {unique_banks}")
        
        # Quarters coverage
        quarters_data = {}
        for record in data:
            quarter = record['quarter']
            quarters_data[quarter] = quarters_data.get(quarter, 0) + 1
            
        print(f"\nQuarters coverage:")
        for quarter in sorted(quarters_data.keys()):
            print(f"  {quarter}: {quarters_data[quarter]:,} data points")
            
        # Concepts coverage
        concepts_data = {}
        for record in data:
            concept = record['concept']
            concepts_data[concept] = concepts_data.get(concept, 0) + 1
            
        print(f"\nTop financial concepts:")
        sorted_concepts = sorted(concepts_data.items(), key=lambda x: x[1], reverse=True)
        for concept, count in sorted_concepts[:10]:
            print(f"  {concept}: {count:,} data points")

def main():
    """Main execution function"""
    # File paths
    banks_csv = "data/raw/top_10_us_banks_with_cik.csv"
    output_csv = "data/processed/top10_banks_forecasting_data.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Initialize extractor
    extractor = SECEdgarExtractor()
    
    try:
        # Extract data for all banks
        all_data, successful_banks = extractor.extract_all_banks(banks_csv)
        
        # Save results
        extractor.save_data(all_data, output_csv)
        
        print(f"\n🎉 EXTRACTION COMPLETE!")
        print(f"✅ Successfully processed {successful_banks} banks")
        print(f"📁 Data saved to: {output_csv}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Extraction interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
