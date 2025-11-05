"""Test BOK API to understand data structure for different statistics."""

import sys
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env.local')

# Test cases: 5 diverse statistics
TEST_CASES = [
    {
        'stat_code': '200Y105',
        'stat_name': '경제활동별 GDP 및 GNI(원계열, 명목, 분기 및 연간)',
        'cycle': 'Q',
        'description': 'GDP 분기별 데이터'
    },
    {
        'stat_code': '404Y014',
        'stat_name': '생산자물가지수(기본분류)',
        'cycle': 'M',
        'description': '생산자물가지수 월간 데이터'
    },
    {
        'stat_code': '102Y002',
        'stat_name': '본원통화 구성내역(평잔, 원계열)',
        'cycle': 'M',
        'description': '본원통화 월간 데이터'
    },
    {
        'stat_code': '731Y001',
        'stat_name': '주요국 통화의 대원화환율',
        'cycle': 'D',
        'description': '환율 일간 데이터'
    },
    {
        'stat_code': '901Y033',
        'stat_name': '전산업생산지수(농림어업제외)',
        'cycle': 'M',
        'description': '산업생산 월간 데이터'
    },
]


def build_url(service: str, auth_key: str, stat_code: str = None, 
              frequency: str = None, start_date: str = None, end_date: str = None) -> str:
    """Build BOK API URL."""
    base = "https://ecos.bok.or.kr/api"
    
    if service == 'StatisticItemList':
        return f"{base}/{service}/{auth_key}/json/kr/1/100/{stat_code}"
    elif service == 'StatisticSearch':
        url = f"{base}/{service}/{auth_key}/json/kr/1/100/{stat_code}/{frequency}/{start_date}/{end_date}"
        # Add item codes as ?/?/?/?
        url += "/?/?/?/?"
        return url
    else:
        return f"{base}/{service}/{auth_key}/json/kr/1/100"


def analyze_item_list(response: Dict[str, Any], stat_code: str) -> Dict[str, Any]:
    """Analyze StatisticItemList response structure."""
    analysis = {
        'stat_code': stat_code,
        'has_items': False,
        'item_count': 0,
        'item_structure': None,
        'sample_items': [],
        'item_codes_available': [],
        'raw_response_keys': list(response.keys())
    }
    
    # Check different possible response structures
    for key in ['StatisticItem', 'StatisticItemList', 'row']:
        if key in response:
            items = response[key]
            if isinstance(items, list) and len(items) > 0:
                analysis['has_items'] = True
                analysis['item_count'] = len(items)
                analysis['sample_items'] = items[:3]
                
                first_item = items[0]
                analysis['item_structure'] = {
                    'keys': list(first_item.keys()),
                    'sample': first_item
                }
                
                # Extract item codes
                item_codes = []
                for item in items:
                    for code_key in ['ITEM_CODE', 'ITEM_CODE1', 'item_code', 'item_code1']:
                        if code_key in item:
                            item_codes.append(item[code_key])
                            break
                analysis['item_codes_available'] = item_codes[:10]
                break
    
    return analysis


def analyze_search_response(response: Dict[str, Any], stat_code: str, cycle: str) -> Dict[str, Any]:
    """Analyze StatisticSearch response structure."""
    analysis = {
        'stat_code': stat_code,
        'cycle': cycle,
        'has_data': False,
        'row_count': 0,
        'column_structure': None,
        'sample_rows': [],
        'date_format': None,
        'value_format': None,
        'raw_response_keys': list(response.keys())
    }
    
    # Check different possible response structures
    for key in ['StatisticSearch', 'row']:
        if key in response:
            data = response[key]
            
            if isinstance(data, list) and len(data) > 0:
                analysis['has_data'] = True
                analysis['row_count'] = len(data)
                analysis['sample_rows'] = data[:5]
                
                first_row = data[0]
                analysis['column_structure'] = {
                    'keys': list(first_row.keys()),
                    'sample': first_row
                }
                
                # Check date and value formats
                for col_key in first_row.keys():
                    key_upper = col_key.upper()
                    if 'DATE' in key_upper or 'TIME' in key_upper or '기간' in col_key:
                        analysis['date_format'] = {
                            'column': col_key,
                            'sample_value': first_row[col_key],
                            'format_type': 'unknown'
                        }
                    if 'VALUE' in key_upper or 'DATA' in key_upper or '값' in col_key or 'DATA_VALUE' in key_upper:
                        analysis['value_format'] = {
                            'column': col_key,
                            'sample_value': first_row[col_key],
                            'type': type(first_row[col_key]).__name__
                        }
                break
            
            elif isinstance(data, dict):
                # Column-oriented structure
                analysis['column_structure'] = {
                    'keys': list(data.keys()),
                    'sample': {k: (str(v)[:100] if isinstance(v, list) else v) for k, v in list(data.items())[:5]}
                }
                if 'TIME' in data or 'DATE' in data:
                    analysis['has_data'] = True
            break
    
    return analysis


def test_statistic(auth_key: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single statistic and analyze its structure."""
    stat_code = test_case['stat_code']
    cycle = test_case['cycle']
    
    print(f"\n{'='*80}")
    print(f"Testing: {test_case['stat_name']}")
    print(f"Stat Code: {stat_code}, Cycle: {cycle}")
    print(f"{'='*80}")
    
    result = {
        'test_case': test_case,
        'item_list': None,
        'search_result': None,
        'errors': []
    }
    
    # Step 1: Get item list
    print("\n1. Getting StatisticItemList...")
    try:
        url = build_url('StatisticItemList', auth_key, stat_code)
        print(f"   URL: {url}")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            item_response = response.json()
        
        result['item_list'] = analyze_item_list(item_response, stat_code)
        print(f"   ✓ Found {result['item_list']['item_count']} items")
        if result['item_list']['item_structure']:
            print(f"   ✓ Structure keys: {result['item_list']['item_structure']['keys']}")
        
        # Save raw response
        result['item_list']['raw_response'] = item_response
        
    except Exception as e:
        error_msg = f"ItemList error: {str(e)}"
        result['errors'].append(error_msg)
        print(f"   ✗ {error_msg}")
    
    time.sleep(0.5)  # Rate limiting
    
    # Step 2: Try to search for data
    print("\n2. Getting StatisticSearch data...")
    try:
        # Determine date range based on cycle
        if cycle == 'D':
            start_date = '20240101'
            end_date = '20241231'
        elif cycle == 'M':
            start_date = '202401'
            end_date = '202412'
        elif cycle == 'Q':
            start_date = '2024Q1'
            end_date = '2024Q4'
        else:
            start_date = '2020'
            end_date = '2024'
        
        url = build_url('StatisticSearch', auth_key, stat_code, cycle, start_date, end_date)
        print(f"   URL: {url}")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            search_response = response.json()
        
        result['search_result'] = analyze_search_response(search_response, stat_code, cycle)
        print(f"   ✓ Found {result['search_result']['row_count']} rows")
        if result['search_result']['column_structure']:
            print(f"   ✓ Structure keys: {result['search_result']['column_structure']['keys']}")
        
        # Save raw response
        result['search_result']['raw_response'] = search_response
            
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        result['errors'].append(error_msg)
        print(f"   ✗ {error_msg}")
    
    return result


def main():
    """Main test function."""
    print("="*80)
    print("BOK API Data Structure Analysis")
    print("="*80)
    
    # Get API key from environment
    auth_key = os.getenv('BOK_API_KEY')
    if not auth_key:
        print("Error: BOK_API_KEY environment variable not set")
        return
    
    # Test each statistic
    all_results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}]")
        try:
            result = test_statistic(auth_key, test_case)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Failed to test {test_case['stat_code']}: {e}")
            all_results.append({
                'test_case': test_case,
                'errors': [str(e)]
            })
        
        if i < len(TEST_CASES):
            time.sleep(1)  # Rate limiting between statistics
    
    # Save results
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'data' / 'api_test_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'bok_api_structure_analysis_{timestamp}.json'
    
    # Save full results (may be large)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Summary
    print("\nSUMMARY:")
    print("-" * 80)
    for result in all_results:
        tc = result['test_case']
        print(f"\n{tc['stat_code']} ({tc['cycle']}): {tc['stat_name']}")
        
        if result.get('item_list'):
            il = result['item_list']
            print(f"  Items: {il['item_count']} available")
            if il['item_structure']:
                print(f"  Item keys: {il['item_structure']['keys']}")
        
        if result.get('search_result'):
            sr = result['search_result']
            print(f"  Data rows: {sr['row_count']}")
            if sr['column_structure']:
                print(f"  Data keys: {sr['column_structure']['keys']}")
                if sr['date_format']:
                    print(f"  Date column: {sr['date_format']['column']} = {sr['date_format']['sample_value']}")
                if sr['value_format']:
                    print(f"  Value column: {sr['value_format']['column']} = {sr['value_format']['sample_value']} (type: {sr['value_format']['type']})")
        
        if result.get('errors'):
            print(f"  Errors: {len(result['errors'])}")
            for err in result['errors']:
                print(f"    - {err}")


if __name__ == '__main__':
    main()
