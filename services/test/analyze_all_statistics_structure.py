"""Test script to analyze data structure for all DFM-selected statistics."""

import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import httpx
import pandas as pd

# Load environment
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / '.env.local')

BOK_API_KEY = os.getenv('BOK_API_KEY')
BOK_BASE_URL = 'https://ecos.bok.or.kr/api'


def call_bok_api(service: str, **params) -> Dict[str, Any]:
    """Call BOK API."""
    # Build URL based on service
    if service == 'StatisticItemList':
        url = f"{BOK_BASE_URL}/{service}/{BOK_API_KEY}/json/kr/1/1000/{params['stat_code']}"
    elif service == 'StatisticSearch':
        stat_code = params['stat_code']
        frequency = params['frequency']
        start_date = params.get('start_date', '2020')
        end_date = params.get('end_date', '2024')
        item_code1 = params.get('item_code1', '')
        
        url = f"{BOK_BASE_URL}/{service}/{BOK_API_KEY}/json/kr/1/1000/{stat_code}/{frequency}/{start_date}/{end_date}/{item_code1}///"
    else:
        raise ValueError(f"Unknown service: {service}")
    
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json()


def analyze_statistic(stat_code: str, cycle: str) -> Dict[str, Any]:
    """Analyze a single statistic's data structure."""
    result = {
        'stat_code': stat_code,
        'cycle': cycle,
        'item_list_structure': {},
        'search_structure': {},
        'errors': []
    }
    
    try:
        # Step 1: Get item list
        item_list_response = call_bok_api('StatisticItemList', stat_code=stat_code)
        
        if 'StatisticItemList' in item_list_response:
            item_data = item_list_response['StatisticItemList']
            rows = item_data.get('row', [])
            
            result['item_list_structure'] = {
                'total_count': item_data.get('list_total_count', 0),
                'num_items': len(rows),
                'sample_item_keys': list(rows[0].keys()) if rows else [],
                'sample_item': rows[0] if rows else None,
                'cycles': list(set(row.get('CYCLE', '') for row in rows)) if rows else []
            }
            
            # Step 2: Get data for first item (if available)
            if rows:
                first_item = rows[0]
                item_code = first_item.get('ITEM_CODE', '')
                item_cycle = first_item.get('CYCLE', cycle)
                
                try:
                    search_response = call_bok_api(
                        'StatisticSearch',
                        stat_code=stat_code,
                        frequency=item_cycle,
                        start_date=first_item.get('START_TIME', '2020'),
                        end_date=first_item.get('END_TIME', '2024'),
                        item_code1=item_code
                    )
                    
                    if 'StatisticSearch' in search_response:
                        search_data = search_response['StatisticSearch']
                        data_rows = search_data.get('row', [])
                        
                        result['search_structure'] = {
                            'total_count': search_data.get('list_total_count', 0),
                            'num_rows': len(data_rows),
                            'sample_row_keys': list(data_rows[0].keys()) if data_rows else [],
                            'sample_row': data_rows[0] if data_rows else None,
                            'item_code_levels': {
                                'has_item_code1': any(row.get('ITEM_CODE1') for row in data_rows),
                                'has_item_code2': any(row.get('ITEM_CODE2') for row in data_rows),
                                'has_item_code3': any(row.get('ITEM_CODE3') for row in data_rows),
                                'has_item_code4': any(row.get('ITEM_CODE4') for row in data_rows),
                            }
                        }
                except Exception as e:
                    result['errors'].append(f"Search API error: {str(e)}")
        else:
            result['errors'].append("No StatisticItemList in response")
            
    except Exception as e:
        result['errors'].append(f"ItemList API error: {str(e)}")
    
    return result


def main():
    """Analyze all DFM-selected statistics."""
    print("=" * 80)
    print("Analyzing BOK API Data Structure for All DFM-Selected Statistics")
    print("=" * 80)
    
    # Load selected statistics
    csv_path = project_root / 'data' / 'bok_statistics_dfm_selected.csv'
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"\n📊 Found {len(df)} statistics to analyze\n")
    
    # Analyze each statistic
    results = []
    for idx, row in df.iterrows():
        stat_code = row['stat_code']
        cycle = row['cycle']
        stat_name = row['stat_name']
        
        print(f"[{idx+1}/{len(df)}] Analyzing: {stat_code} ({cycle}) - {stat_name}")
        
        result = analyze_statistic(stat_code, cycle)
        results.append(result)
        
        if result['errors']:
            print(f"  ⚠️  Errors: {', '.join(result['errors'])}")
        else:
            print(f"  ✓ Items: {result['item_list_structure'].get('num_items', 0)}, "
                  f"Data rows: {result['search_structure'].get('num_rows', 0)}")
    
    # Save results
    output_dir = project_root / 'data' / 'api_structure_analysis'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f'structure_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Analyze common patterns
    print("\n" + "=" * 80)
    print("Structure Analysis Summary")
    print("=" * 80)
    
    # Collect all unique keys
    all_item_keys = set()
    all_row_keys = set()
    item_code_usage = {'item_code1': 0, 'item_code2': 0, 'item_code3': 0, 'item_code4': 0}
    
    for result in results:
        if result['item_list_structure'].get('sample_item_keys'):
            all_item_keys.update(result['item_list_structure']['sample_item_keys'])
        if result['search_structure'].get('sample_row_keys'):
            all_row_keys.update(result['search_structure']['sample_row_keys'])
        if result['search_structure'].get('item_code_levels'):
            levels = result['search_structure']['item_code_levels']
            for key in item_code_usage:
                if levels.get(f'has_{key}'):
                    item_code_usage[key] += 1
    
    print(f"\nItem List Keys (all statistics):")
    for key in sorted(all_item_keys):
        print(f"  - {key}")
    
    print(f"\nSearch Result Keys (all statistics):")
    for key in sorted(all_row_keys):
        print(f"  - {key}")
    
    print(f"\nItem Code Usage:")
    for key, count in item_code_usage.items():
        print(f"  - {key}: {count}/{len(results)} statistics")
    
    # Check for variations
    print(f"\nStructure Variations:")
    item_key_counts = {}
    row_key_counts = {}
    
    for result in results:
        item_keys = tuple(sorted(result['item_list_structure'].get('sample_item_keys', [])))
        row_keys = tuple(sorted(result['search_structure'].get('sample_row_keys', [])))
        item_key_counts[item_keys] = item_key_counts.get(item_keys, 0) + 1
        row_key_counts[row_keys] = row_key_counts.get(row_keys, 0) + 1
    
    print(f"  Item List key variations: {len(item_key_counts)}")
    print(f"  Search Result key variations: {len(row_key_counts)}")
    
    if len(item_key_counts) > 1:
        print(f"\n  ⚠️  Item List structures vary:")
        for keys, count in sorted(item_key_counts.items(), key=lambda x: -x[1]):
            print(f"    {count} statistics: {keys[:5]}...")
    
    if len(row_key_counts) > 1:
        print(f"\n  ⚠️  Search Result structures vary:")
        for keys, count in sorted(row_key_counts.items(), key=lambda x: -x[1]):
            print(f"    {count} statistics: {keys[:5]}...")


if __name__ == '__main__':
    main()

