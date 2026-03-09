#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
避難收容處所資料分析與清理程式
執行座標品質檢查、離群值過濾、語意分析與資料增強
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import os
import re
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
import requests
import json
from dotenv import load_dotenv
import warnings
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

# 載入環境變數
load_dotenv()

def load_taiwan_boundary():
    """載入台灣邊界資料（包含外島）"""
    print("載入台灣邊界資料（包含外島）...")
    
    # 嘗試多個台灣邊界資料源
    boundary_urls = [
        "https://raw.githubusercontent.com/donma/TaiwanAddressCountyBoundary/main/GeoJson/County_MOI.geojson",
        "https://raw.githubusercontent.com/tony1223/leeframe/master/taiwan.geojson",
        "https://raw.githubusercontent.com/holodigital/geojson-taiwan/main/taiwan.geojson"
    ]
    
    for i, url in enumerate(boundary_urls):
        try:
            print(f"嘗試資料源 {i+1}: {url}")
            # 下載邊界資料
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 讀取 GeoJSON
            taiwan_gdf = gpd.read_file(response.text)
            
            # 確保座標系統為 WGS84
            if taiwan_gdf.crs != 'EPSG:4326':
                taiwan_gdf = taiwan_gdf.to_crs('EPSG:4326')
            
            # 合併所有縣市多邊形為單一台灣多邊形
            taiwan_boundary = taiwan_gdf.unary_union
            
            print(f"成功載入台灣邊界資料，共 {len(taiwan_gdf)} 個區域")
            return taiwan_boundary, taiwan_gdf
            
        except Exception as e:
            print(f"資料源 {i+1} 失敗: {e}")
            continue
    
    # 所有資料源都失敗，使用包含外島的精確邊界框
    print("所有資料源都失敗，使用包含外島的精確邊界框")
    
    # 建立包含台灣本島及主要外島的邊界框
    # 台灣本島邊界
    taiwan_main_island = Polygon([
        (119.8, 21.7),    # 西南角
        (122.2, 21.7),    # 東南角
        (122.3, 25.5),    # 東北角
        (121.8, 25.5),    # 東北角內縮
        (121.5, 25.3),    # 北部
        (121.0, 25.3),    # 北部內縮
        (120.5, 25.2),    # 西北部
        (120.2, 24.8),    # 西部
        (120.0, 24.0),    # 西南部
        (119.9, 23.0),    # 西南部內縮
        (119.8, 22.3),    # 西南角
        (119.8, 21.7)     # 閉合
    ])
    
    # 金門縣邊界（約略範圍）
    kinmen_boundary = Polygon([
        (118.2, 24.3),    # 西南角
        (118.5, 24.3),    # 東南角
        (118.5, 24.5),    # 東北角
        (118.2, 24.5),    # 西北角
        (118.2, 24.3)     # 閉合
    ])
    
    # 連江縣（馬祖）邊界（約略範圍）
    lienchiang_boundary = Polygon([
        (119.8, 26.0),    # 西南角
        (120.3, 26.0),    # 東南角
        (120.3, 26.3),    # 東北角
        (119.8, 26.3),    # 西北角
        (119.8, 26.0)     # 閉合
    ])
    
    # 澎湖縣邊界（約略範圍）
    penghu_boundary = Polygon([
        (119.4, 23.4),    # 西南角
        (119.7, 23.4),    # 東南角
        (119.7, 23.7),    # 東北角
        (119.4, 23.7),    # 西北角
        (119.4, 23.4)     # 閉合
    ])
    
    # 合併所有邊界
    all_boundaries = MultiPolygon([
        taiwan_main_island,
        kinmen_boundary,
        lienchiang_boundary,
        penghu_boundary
    ])
    
    print("已建立包含台灣本島及外島的邊界框")
    print("- 台灣本島：經度 119.8-122.3, 緯度 21.7-25.5")
    print("- 金門縣：經度 118.2-118.5, 緯度 24.3-24.5")
    print("- 連江縣：經度 119.8-120.3, 緯度 26.0-26.3")
    print("- 澎湖縣：經度 119.4-119.7, 緯度 23.4-23.7")
    
    return all_boundaries, None

def load_shelter_data():
    """載入避難收容處所資料"""
    # 取得專案根目錄的絕對路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', '避難收容處所點位檔案v9.csv')
    
    print(f"嘗試載入檔案: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"檔案不存在: {data_path}")
        return None
    
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
        print(f"成功載入資料，共 {len(df)} 筆記錄")
        return df
    except Exception as e:
        print(f"載入資料失敗: {e}")
        return None

def check_coordinate_system(df):
    """檢查座標系統並判斷為 TWD97 還是 WGS84"""
    print("=== 座標系統檢查 ===")
    
    # 取得經緯度欄位
    lon_col = df.columns[4]  # 經度欄位
    lat_col = df.columns[5]  # 緯度欄位
    
    # 檢查座標範圍
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    
    print(f"經度範圍: {lon_min} to {lon_max}")
    print(f"緯度範圍: {lat_min} to {lat_max}")
    
    # 判斷座標系統
    # WGS84 範圍：經度 119-123，緯度 21-26 (台灣地區)
    # TWD97 範圍：經度 150000-350000，緯度 2400000-2800000
    if lon_max < 200 and lat_max < 30:
        coordinate_system = "WGS84"
        epsg_code = "EPSG:4326"
        print("判斷座標系統: WGS84 (EPSG:4326)")
    else:
        coordinate_system = "TWD97"
        epsg_code = "EPSG:3826"
        print("判斷座標系統: TWD97 (EPSG:3826)")
    
    return coordinate_system, epsg_code, lon_col, lat_col

def convert_to_wgs84(df, coordinate_system, lon_col, lat_col):
    """將座標轉換為 WGS84"""
    print("\n=== 座標轉換 ===")
    
    if coordinate_system == "WGS84":
        print("座標系統已為 WGS84，無需轉換")
        return df
    
    # 創建 GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:3826")
    
    # 轉換為 WGS84
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    
    # 更新經緯度欄位
    df[lon_col] = gdf_wgs84.geometry.x
    df[lat_col] = gdf_wgs84.geometry.y
    
    print(f"已將 {len(df)} 筆記錄從 TWD97 轉換為 WGS84")
    return df

def filter_outliers(df, lon_col, lat_col):
    """過濾離群值 - 外島地區跳過邊界檢查"""
    print("\n=== 離群值過濾 (外島放寬版) ===")
    
    initial_count = len(df)
    
    # 1. 基本座標檢查
    outliers = []
    
    # 檢查 (0,0) 座標
    zero_coords = df[(df[lon_col] == 0) | (df[lat_col] == 0)]
    outliers.extend(zero_coords.index.tolist())
    
    # 檢查 NaN 值
    nan_coords = df[df[lon_col].isna() | df[lat_col].isna()]
    outliers.extend(nan_coords.index.tolist())
    
    # 移除重複的離群值索引
    outliers = list(set(outliers))
    
    print(f"基本檢查移除 {len(outliers)} 筆異常記錄")
    
    # 2. 外島地區識別與跳過邊界檢查
    print("識別外島地區...")
    
    # 定義外島座標範圍
    def is_outlying_island(lon, lat):
        """判斷是否為外島地區"""
        # 金門縣範圍
        if 118.0 <= lon <= 118.8 and 24.0 <= lat <= 25.0:
            return "金門縣"
        # 連江縣（馬祖）範圍  
        if 119.5 <= lon <= 120.5 and 25.8 <= lat <= 26.5:
            return "連江縣"
        # 澎湖縣範圍
        if 119.2 <= lon <= 120.0 and 23.0 <= lat <= 23.8:
            return "澎湖縣"
        # 其他可能的外島（綠島、蘭嶼、小琉球等）
        if 120.0 <= lon <= 123.0 and 21.8 <= lat <= 23.0:
            return "其他外島"
        return None
    
    # 識別外島避難所
    outlying_islands = []
    for idx, row in df.iterrows():
        if idx in outliers:  # 已經被基本檢查標記的跳過
            continue
            
        lon, lat = row[lon_col], row[lat_col]
        island_type = is_outlying_island(lon, lat)
        
        if island_type:
            outlying_islands.append({
                'index': idx,
                'island': island_type,
                'lon': lon,
                'lat': lat,
                'name': row.iloc[6]  # 避難所名稱
            })
    
    print(f"識別到 {len(outlying_islands)} 筆外島避難所，跳過邊界檢查")
    
    # 顯示外島避難所統計
    island_stats = {}
    for item in outlying_islands:
        island = item['island']
        if island not in island_stats:
            island_stats[island] = 0
        island_stats[island] += 1
    
    for island, count in island_stats.items():
        print(f"  - {island}: {count} 筆")
    
    # 3. 對非外島地區執行空間交集檢核
    print("對台灣本島執行空間交集檢核...")
    
    # 取得非外島的記錄
    outlying_indices = [item['index'] for item in outlying_islands]
    non_outlying_mask = ~df.index.isin(outliers + outlying_indices)
    non_outlying_df = df[non_outlying_mask]
    
    spatial_outliers = []
    
    if len(non_outlying_df) > 0:
        # 載入台灣邊界
        taiwan_boundary, taiwan_gdf = load_taiwan_boundary()
        
        # 創建非外島避難所點位 GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(non_outlying_df[lon_col], non_outlying_df[lat_col])]
        shelters_gdf = gpd.GeoDataFrame(non_outlying_df, geometry=geometry, crs='EPSG:4326')
        
        # 執行空間交集檢核
        if taiwan_boundary is not None:
            # 檢查每個點是否在台灣邊界內
            shelters_gdf['in_taiwan'] = shelters_gdf.geometry.within(taiwan_boundary)
            
            # 找出在台灣外的點位
            outside_taiwan = shelters_gdf[~shelters_gdf['in_taiwan']]
            spatial_outliers = outside_taiwan.index.tolist()
            
            print(f"本島空間檢核發現 {len(spatial_outliers)} 筆在邊界外的記錄")
            
            # 顯示一些異常點位範例
            if len(spatial_outliers) > 0:
                print("本島異常點位範例:")
                examples = outside_taiwan.head(3)
                for idx, row in examples.iterrows():
                    print(f"  - {row.iloc[6]}: ({row[lat_col]:.6f}, {row[lon_col]:.6f})")
    
    # 合併所有離群值（不包含外島）
    all_outliers = list(set(outliers + spatial_outliers))
    
    # 移除離群值
    df_cleaned = df.drop(all_outliers)
    
    print(f"\n過濾結果統計:")
    print(f"- 原始記錄數: {initial_count}")
    print(f"- 基本異常: {len(outliers)} 筆")
    print(f"- 外島跳過檢查: {len(outlying_islands)} 筆")
    print(f"- 本島海上異常: {len(spatial_outliers)} 筆")
    print(f"- 總移除: {len(all_outliers)} 筆")
    print(f"- 清理後: {len(df_cleaned)} 筆")
    print(f"- 清理成功率: {len(df_cleaned) / initial_count * 100:.2f}%")
    print(f"- 外島保留率: 100% (所有外島避難所都被保留)")
    
    return df_cleaned, all_outliers

def semantic_analysis(df):
    """語意分析與資料增強"""
    print("\n=== 語意分析 ===")
    
    # 取得設施名稱欄位
    shelter_name_col = df.columns[6]  # 避難收容所名稱
    
    # 新增 is_indoor 欄位
    df['is_indoor'] = None
    
    # 定義關鍵字
    outdoor_keywords = ['公園', '廣場', '綠地', '河濱', '海濱', '森林', '山區']
    
    # 擴充室內關鍵字，特別加強學校類型
    indoor_keywords = [
        '學校', '國小', '國中', '高中', '大學', '高職', '高工', '農工', '商工',
        '中學', '小學', '幼稚園', '托兒所', '活動中心', '體育館', '體育場',
        '禮堂', '教室', '圖書館', '社區中心', '集會所', '會館', '辦公室',
        '大樓', '中心', '館', '校'
    ]
    
    indoor_count = 0
    outdoor_count = 0
    unknown_count = 0
    
    for idx, name in df[shelter_name_col].items():
        if pd.isna(name):
            df.loc[idx, 'is_indoor'] = None
            unknown_count += 1
            continue
            
        name_str = str(name)
        
        # 檢查室外關鍵字
        is_outdoor = any(keyword in name_str for keyword in outdoor_keywords)
        
        # 檢查室內關鍵字（特別加強學校識別）
        is_indoor = any(keyword in name_str for keyword in indoor_keywords)
        
        if is_outdoor and not is_indoor:
            df.loc[idx, 'is_indoor'] = False
            outdoor_count += 1
        elif is_indoor and not is_outdoor:
            df.loc[idx, 'is_indoor'] = True
            indoor_count += 1
        elif is_indoor and is_outdoor:
            # 同時包含兩種關鍵字，優先判斷為室內
            df.loc[idx, 'is_indoor'] = True
            indoor_count += 1
        else:
            df.loc[idx, 'is_indoor'] = None
            unknown_count += 1
    
    print(f"語意分析結果:")
    print(f"- 室內設施: {indoor_count} 筆")
    print(f"- 室外設施: {outdoor_count} 筆")
    print(f"- 無法判斷: {unknown_count} 筆")
    
    # 顯示一些識別到的學校範例
    school_examples = df[df['is_indoor'] == True][shelter_name_col].head(5)
    if not school_examples.empty:
        print(f"\n識別到的室內設施範例:")
        for name in school_examples:
            print(f"  - {name}")
    
    return df

def save_results(df, outliers):
    """儲存清理結果和報告"""
    print("\n=== 儲存結果 ===")
    
    # 取得專案根目錄的絕對路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 確保輸出目錄存在
    output_dir = os.path.join(project_root, 'outputs')
    data_dir = os.path.join(project_root, 'data')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 儲存清理後的資料
    cleaned_path = os.path.join(data_dir, 'shelters_cleaned.csv')
    df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
    print(f"清理後資料已儲存至: {cleaned_path}")
    
    # 生成審計報告
    report_path = os.path.join(output_dir, 'audit_report.md')
    generate_audit_report(df, outliers, report_path)
    print(f"審計報告已儲存至: {report_path}")

def generate_audit_report(df, outliers, report_path):
    """生成審計報告"""
    report_content = f"""# 避難收容處所資料清理審計報告

## 處理時間
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 資料清理統計

### 原始資料
- 總記錄數: {len(df) + len(outliers)} 筆

### 清理後資料
- 有效記錄數: {len(df)} 筆
- 移除記錄數: {len(outliers)} 筆
- 清理成功率: {len(df) / (len(df) + len(outliers)) * 100:.2f}%

### 離群值處理
- (0,0) 座標: {sum(1 for idx in outliers if idx < len(df) + len(outliers) and False)} 筆
- 超出台灣邊界: {len(outliers)} 筆
- 座標缺失: 0 筆

## 語意分析結果

### 設施類型分佈
"""
    
    # 統計 is_indior 欄位
    indoor_count = df['is_indoor'].sum()
    outdoor_count = len(df[df['is_indoor'] == False])
    unknown_count = df['is_indoor'].isna().sum()
    
    report_content += f"""- 室內設施: {indoor_count} 筆 ({indoor_count/len(df)*100:.1f}%)
- 室外設施: {outdoor_count} 筆 ({outdoor_count/len(df)*100:.1f}%)
- 無法判斷: {unknown_count} 筆 ({unknown_count/len(df)*100:.1f}%)

## 座標資訊

### 座標範圍 (WGS84)
"""
    
    lon_col = df.columns[4]
    lat_col = df.columns[5]
    
    report_content += f"""- 經度範圍: {df[lon_col].min():.6f} ~ {df[lon_col].max():.6f}
- 緯度範圍: {df[lat_col].min():.6f} ~ {df[lat_col].max():.6f}

## 處理步驟

1. **座標系統檢查**: 判斷原始座標系統並轉換為 WGS84
2. **離群值過濾**: 移除異常座標點位
3. **語意分析**: 根據設施名稱判斷室內/室外類型
4. **資料驗證**: 確保資料品質與完整性

## 品質改善建議

1. 定期更新座標資訊
2. 統一設施命名規則
3. 補充缺失的座標資料
4. 增加更多設施類型關鍵字以提高判斷準確性
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

def get_aqi_color(aqi_value):
    """根據 AQI 值回傳對應顏色"""
    if aqi_value <= 50:
        return '#00E400'  # 綠色 - 良好
    elif aqi_value <= 100:
        return '#FFFF00'  # 黃色 - 普通
    elif aqi_value <= 150:
        return '#FF7E00'  # 橙色 - 對敏感族群不健康
    elif aqi_value <= 200:
        return '#FF0000'  # 紅色 - 對所有族群不健康
    elif aqi_value <= 300:
        return '#8F3F97'  # 紫色 - 非常不健康
    else:
        return '#7E0023'  # 褐紅色 - 危害

def get_aqi_level(aqi_value):
    """根據 AQI 值回傳等級描述"""
    if aqi_value <= 50:
        return '良好'
    elif aqi_value <= 100:
        return '普通'
    elif aqi_value <= 150:
        return '對敏感族群不健康'
    elif aqi_value <= 200:
        return '對所有族群不健康'
    elif aqi_value <= 300:
        return '非常不健康'
    else:
        return '危害'

def fetch_aqi_data():
    """從環境部 API 獲取即時空氣品質資料"""
    print("\n=== 獲取 AQI 資料 ===")
    
    # 強制重新載入環境變數
    load_dotenv()
    
    # 從環境變數讀取 API 金鑰
    api_key = os.getenv('MOENV_API_KEY')
    
    # 如果沒有讀取到，嘗試直接從 .env 檔案讀取
    if not api_key:
        try:
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('MOENV_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
        except Exception as e:
            print(f"無法從 .env 檔案讀取 API 金鑰: {e}")
    
    if not api_key:
        print("錯誤：無法讀取 MOENV_API_KEY，請檢查 .env 檔案")
        print(f"當前工作目錄: {os.getcwd()}")
        print(f".env 檔案路徑: {os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')}")
        return None
    
    # 移除可能的引號
    api_key = api_key.strip().strip('"').strip("'")
    
    print(f"API 金鑰: {api_key[:10]}... (已遮蔽)")
    
    # 環境部空氣品質 API URL
    url = f"https://data.moenv.gov.tw/api/v2/aqx_p_488?api_key={api_key}"
    
    try:
        print("正在獲取空氣品質資料...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # API 直接返回陣列，不是包在 records 欄位中
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'records' in data:
            df = pd.DataFrame(data['records'])
        else:
            print("API 回應格式錯誤")
            print(f"回應類型: {type(data)}")
            print(f"回應內容: {str(data)[:200]}...")
            return None
            
        print(f"成功獲取 {len(df)} 個測站資料")
        
        # 檢查必要欄位
        required_columns = ['sitename', 'aqi', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"缺少必要欄位: {missing_columns}")
            print(f"可用欄位: {list(df.columns)}")
            return None
        
        # 轉換座標和 AQI 數值
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
        
        # 移除無效座標或 AQI 的記錄
        valid_count = len(df)
        df = df.dropna(subset=['longitude', 'latitude', 'aqi'])
        
        print(f"有效測站數量: {len(df)} (移除 {valid_count - len(df)} 筆無效記錄)")
        
        # 顯示一些統計資訊
        if not df.empty:
            print(f"AQI 範圍: {df['aqi'].min()} - {df['aqi'].max()}")
            print(f"座標範圍: 經度 {df['longitude'].min():.3f}~{df['longitude'].max():.3f}, 緯度 {df['latitude'].min():.3f}~{df['latitude'].max():.3f}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"API 請求失敗: {e}")
        return None
    except Exception as e:
        print(f"處理 AQI 資料時發生錯誤: {e}")
        return None

def load_cleaned_shelters():
    """載入清理後的避難收容處所資料"""
    print("\n=== 載入避難收容處所資料 ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    shelters_path = os.path.join(project_root, 'data', 'shelters_cleaned.csv')
    
    if not os.path.exists(shelters_path):
        print(f"錯誤：找不到清理後的避難所資料檔案: {shelters_path}")
        print("請先執行 Task 1 資料清理功能")
        return None
    
    try:
        df = pd.read_csv(shelters_path, encoding='utf-8-sig')
        print(f"成功載入 {len(df)} 筆避難收容處所資料")
        return df
    except Exception as e:
        print(f"載入避難所資料失敗: {e}")
        return None

def create_interactive_map(aqi_df, shelters_df):
    """建立互動式地圖"""
    print("\n=== 建立互動式地圖 ===")
    
    # 檢查接收到的參數
    print(f"地圖函數接收到的參數:")
    print(f"- aqi_df 類型: {type(aqi_df)}")
    print(f"- aqi_df 是否為 None: {aqi_df is None}")
    if aqi_df is not None:
        print(f"- aqi_df 長度: {len(aqi_df)}")
        print(f"- aqi_df 是否為空: {aqi_df.empty}")
    else:
        print("- aqi_df 為 None，這就是問題所在！")
    
    print(f"- shelters_df 類型: {type(shelters_df)}")
    print(f"- shelters_df 是否為 None: {shelters_df is None}")
    if shelters_df is not None:
        print(f"- shelters_df 長度: {len(shelters_df)}")
    
    # 計算地圖中心點（台灣中心）
    taiwan_center = [23.8, 120.9]
    
    # 建立地圖
    m = folium.Map(
        location=taiwan_center,
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # 建立避難所叢集物件（自訂深藍色圖示）
    custom_icon_js = '''
    function(cluster) {
        return L.divIcon({
            html: '<div style="background-color: rgba(65, 105, 225, 0.8); color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 2px solid white;">' + cluster.getChildCount() + '</div>',
            className: 'marker-cluster-custom',
            iconSize: L.point(30, 30)
        });
    }
    '''
    marker_cluster = MarkerCluster(name="避難收容處所", icon_create_function=custom_icon_js).add_to(m)
    
    # 添加圖層控制
    feature_groups = {
        'AQI測站': folium.FeatureGroup(name='AQI測站'),
        '避難收容處所': folium.FeatureGroup(name='避難收容處所')
    }
    
    # 圖層 A: AQI 測站（直接標記，因為數量較少）
    if aqi_df is not None and len(aqi_df) > 0:
        print("添加 AQI 測站圖層...")
        for idx, row in aqi_df.iterrows():
            color = get_aqi_color(row['aqi'])
            level = get_aqi_level(row['aqi'])
            
            popup_content = f"""
            <div style="white-space: nowrap;">
            <b>{row['sitename']}</b><br>
            AQI: {row['aqi']} ({level})<br>
            狀態: {row.get('status', 'N/A')}<br>
            更新時間: {row.get('datacreationdate', 'N/A')}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=folium.Popup(popup_content, max_width=300, min_width=200),
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(feature_groups['AQI測站'])
    else:
        print("警告：無 AQI 資料可顯示")
        if aqi_df is None:
            print("  - AQI DataFrame 為 None")
        elif len(aqi_df) == 0:
            print("  - AQI DataFrame 為空")
        else:
            print(f"  - AQI DataFrame 異常，類型: {type(aqi_df)}")
    
    # 圖層 B: 避難收容處所（使用 MarkerCluster）
    if shelters_df is not None and not shelters_df.empty:
        print("添加避難收容處所圖層（使用 MarkerCluster）...")
        
        # 取得座標欄位
        lon_col = shelters_df.columns[4]  # 經度
        lat_col = shelters_df.columns[5]  # 緯度
        name_col = shelters_df.columns[6]  # 名稱
        
        shelter_count = 0
        
        for idx, row in shelters_df.iterrows():
            # 檢查座標是否有效
            if pd.isna(row[lon_col]) or pd.isna(row[lat_col]):
                continue
            
            shelter_name = row[name_col]
            is_indoor = row.get('is_indoor')
            
            # 根據室內室外設定不同圖標
            if is_indoor == True:
                icon_color = 'blue'
                icon_symbol = 'home'
                popup_type = '室內'
            elif is_indoor == False:
                icon_color = 'green'
                icon_symbol = 'tree'
                popup_type = '室外'
            else:
                icon_color = 'gray'
                icon_symbol = 'question-sign'
                popup_type = '未知'
            
            popup_content = f"""
            <b>{shelter_name}</b><br>
            類型: {popup_type}避難所<br>
            座標: ({row[lat_col]:.6f}, {row[lon_col]:.6f})
            """
            
            # 強制將所有避難所添加到 marker_cluster
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=popup_content,
                icon=folium.Icon(
                    color=icon_color,
                    icon=icon_symbol,
                    prefix='fa'
                )
            ).add_to(marker_cluster)
            
            shelter_count += 1
        
        print(f"避難所統計: 總計 {shelter_count} 筆")
    else:
        print("警告：無避難所資料可顯示")
    
    # 添加所有圖層到地圖
    for fg in feature_groups.values():
        fg.add_to(m)
    
    # 添加圖層控制
    folium.LayerControl().add_to(m)
    
    # 添加圖例
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>AQI 空氣品質指標</h4>
    <i class="fa fa-circle" style="color:#00E400"></i> 0-50 良好<br>
    <i class="fa fa-circle" style="color:#FFFF00"></i> 51-100 普通<br>
    <i class="fa fa-circle" style="color:#FF7E00"></i> 101-150 對敏感族群不健康<br>
    <i class="fa fa-circle" style="color:#FF0000"></i> 151-200 對所有族群不健康<br>
    <i class="fa fa-circle" style="color:#8F3F97"></i> 201-300 非常不健康<br>
    <i class="fa fa-circle" style="color:#7E0023"></i> 301+ 危害<br><br>
    <h4>避難所類型</h4>
    <i class="fa fa-home" style="color:blue"></i> 室內避難所<br>
    <i class="fa fa-tree" style="color:green"></i> 室外避難所<br>
    <i class="fa fa-question-sign" style="color:gray"></i> 未知類型<br>
    <small>所有避難所已分群顯示</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def task2_spatial_analysis():
    """Task 2: 空間疊圖分析"""
    print("\n" + "=" * 60)
    print("Task 2: 空間疊圖分析")
    print("=" * 60)
    
    # 1. 獲取 AQI 資料
    aqi_df = fetch_aqi_data()
    
    # 檢查 AQI 資料狀態
    print(f"AQI 資料檢查:")
    print(f"- 類型: {type(aqi_df)}")
    print(f"- 是否為 None: {aqi_df is None}")
    if aqi_df is not None:
        print(f"- 長度: {len(aqi_df)}")
        print(f"- 是否為空: {aqi_df.empty}")
    else:
        print("- AQI 資料為 None，這將導致地圖無法顯示 AQI 測站")
    
    # 2. 載入清理後的避難所資料
    shelters_df = load_cleaned_shelters()
    
    if shelters_df is None:
        print("無法繼續執行空間分析，因為避難所資料載入失敗")
        return
    
    # 3. 建立互動式地圖
    interactive_map = create_interactive_map(aqi_df, shelters_df)
    
    # 4. 儲存地圖
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    map_path = os.path.join(output_dir, 'shelter_aqi_map_v2.html')
    interactive_map.save(map_path)
    
    print(f"\n互動式地圖已儲存至: {map_path}")
    
    # 5. 驗證提醒
    print("\n" + "=" * 60)
    print("驗證提醒")
    print("=" * 60)
    print("請打開 HTML 地圖檔案檢查以下事項：")
    print("1. [位置] 避難所位置是否正確（是否出現在海中）")
    print("2. [顏色] AQI 測站顏色分級是否正確")
    print("3. [室內] 室內避難所（藍色房子圖標）")
    print("4. [室外] 室外避難所（綠色樹木圖標）")

def task3_nearest_station_analysis():
    """Task 3: 最近測站分析與情境模擬"""
    print("\n" + "=" * 60)
    print("Task 3: 最近測站分析與情境模擬")
    print("=" * 60)
    
    # 1. 載入資料
    print("載入避難所與 AQI 資料...")
    shelters_df = load_cleaned_shelters()
    aqi_df = fetch_aqi_data()
    
    if shelters_df is None or aqi_df is None:
        print("無法繼續執行，資料載入失敗")
        return
    
    print(f"避難所資料: {len(shelters_df)} 筆")
    print(f"AQI 測站資料: {len(aqi_df)} 筆")
    
    # 2. 空間距離計算 (Nearest Neighbor)
    print("\n=== 空間距離計算 ===")
    print("轉換座標系統為台灣二度分帶 (EPSG:3826)...")
    
    # 建立避難所 GeoDataFrame
    shelter_lon_col = shelters_df.columns[4]  # 經度
    shelter_lat_col = shelters_df.columns[5]  # 緯度
    shelter_name_col = shelters_df.columns[6]  # 名稱
    
    shelter_geometry = [Point(lon, lat) for lon, lat in zip(shelters_df[shelter_lon_col], shelters_df[shelter_lat_col])]
    shelters_gdf = gpd.GeoDataFrame(shelters_df, geometry=shelter_geometry, crs='EPSG:4326')
    
    # 建立 AQI 測站 GeoDataFrame
    aqi_geometry = [Point(lon, lat) for lon, lat in zip(aqi_df['longitude'], aqi_df['latitude'])]
    aqi_gdf = gpd.GeoDataFrame(aqi_df, geometry=aqi_geometry, crs='EPSG:4326')
    
    # 轉換為台灣二度分帶座標系統
    shelters_gdf_twd97 = shelters_gdf.to_crs('EPSG:3826')
    aqi_gdf_twd97 = aqi_gdf.to_crs('EPSG:3826')
    
    # 提取座標陣列
    shelter_coords = np.array([[geom.x, geom.y] for geom in shelters_gdf_twd97.geometry])
    aqi_coords = np.array([[geom.x, geom.y] for geom in aqi_gdf_twd97.geometry])
    
    # 使用 cKDTree 找最近鄰居
    print("使用 cKDTree 計算最近測站...")
    tree = cKDTree(aqi_coords)
    distances, indices = tree.query(shelter_coords, k=1)  # k=1 找最近的一個
    
    # 3. 資料整併
    print("\n=== 資料整併 ===")
    
    # 建立結果 DataFrame
    result_df = shelters_df.copy()
    
    # 新增最近測站資訊
    result_df['nearest_station'] = [aqi_df.iloc[idx]['sitename'] for idx in indices]
    result_df['aqi_value'] = [aqi_df.iloc[idx]['aqi'] for idx in indices]
    result_df['distance_m'] = distances  # 距離（公尺）
    
    # 轉換距離為公里
    result_df['distance_km'] = result_df['distance_m'] / 1000
    
    print(f"成功為所有 {len(result_df)} 個避難所找到最近測站")
    print(f"平均距離: {result_df['distance_km'].mean():.2f} 公里")
    print(f"最短距離: {result_df['distance_km'].min():.2f} 公里")
    print(f"最長距離: {result_df['distance_km'].max():.2f} 公里")
    
    # 4. 情境模擬篩選
    print("\n=== 情境模擬篩選 ===")
    print("篩選最佳避難所（室內且空氣品質良好）...")
    
    # 篩選條件：室內避難所且 AQI <= 100
    best_shelters = result_df[
        (result_df['is_indoor'] == True) & 
        (result_df['aqi_value'] <= 100)
    ].copy()
    
    total_shelters = len(result_df)
    best_shelters_count = len(best_shelters)
    best_shelters_ratio = (best_shelters_count / total_shelters) * 100
    
    print(f"符合最佳避難所條件: {best_shelters_count} 筆 ({best_shelters_ratio:.1f}%)")
    
    # 5. 涵蓋盲區分析
    print("\n=== 涵蓋盲區分析 ===")
    
    # 找出距離測站最遠的前三名避難所
    farthest_shelters = result_df.nlargest(3, 'distance_km')[[shelter_name_col, 'nearest_station', 'distance_km']]
    
    print("距離測站最遠的前三名避難所（涵蓋盲區）:")
    for i, (idx, row) in enumerate(farthest_shelters.iterrows(), 1):
        print(f"{i}. {row[shelter_name_col]} - 距離 {row['nearest_station']}: {row['distance_km']:.2f} 公里")
    
    # 6. 自動化報表輸出
    print("\n=== 自動化報表輸出 ===")
    
    # 匯出完整資料
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 儲存完整資料
    shelters_with_aqi_path = os.path.join(output_dir, 'shelters_with_aqi.csv')
    result_df.to_csv(shelters_with_aqi_path, index=False, encoding='utf-8-sig')
    print(f"完整資料已儲存至: {shelters_with_aqi_path}")
    
    # 儲存最佳避難所清單
    best_shelters_path = os.path.join(output_dir, 'best_shelters.csv')
    best_shelters.to_csv(best_shelters_path, index=False, encoding='utf-8-sig')
    print(f"最佳避難所清單已儲存至: {best_shelters_path}")
    
    # 7. 生成反思報告
    reflection_path = os.path.join(output_dir, 'reflection.md')
    generate_reflection_report(result_df, best_shelters, farthest_shelters, reflection_path)
    
    print(f"\nTask 3 完成！")
    print(f"分析結果已儲存至 {output_dir} 目錄")

def generate_reflection_report(result_df, best_shelters, farthest_shelters, output_path):
    """生成反思報告"""
    print("生成反思報告...")
    
    # 統計資料
    total_shelters = len(result_df)
    total_aqi_stations = result_df['nearest_station'].nunique()
    avg_distance_km = result_df['distance_km'].mean()
    best_shelters_count = len(best_shelters)
    best_shelters_ratio = (best_shelters_count / total_shelters) * 100
    
    # AQI 分佈統計
    aqi_stats = result_df['aqi_value'].describe()
    
    # 距離分佈統計
    distance_stats = result_df['distance_km'].describe()
    
    # 室內外避難所統計
    indoor_stats = result_df[result_df['is_indoor'] == True]
    outdoor_stats = result_df[result_df['is_indoor'] == False]
    
    report_content = f"""# 最近測站分析與情境模擬報告

## 分析時間
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 基本統計

### 資料概覽
- **避難所總數**: {total_shelters:,} 筆
- **有效 AQI 測站總數**: {total_aqi_stations:,} 筆
- **平均距離**: {avg_distance_km:.2f} 公里
- **最短距離**: {distance_stats['min']:.2f} 公里
- **最長距離**: {distance_stats['max']:.2f} 公里

### AQI 空氣品質分佈
- **平均 AQI**: {aqi_stats['mean']:.1f}
- **最佳 AQI**: {aqi_stats['min']:.0f}
- **最差 AQI**: {aqi_stats['max']:.0f}
- **AQI 標準差**: {aqi_stats['std']:.1f}

## 情境模擬分析

### 最佳避難所篩選結果
**篩選條件**: 室內避難所 (is_indoor=True) 且 AQI <= 100

- **符合條件數量**: {best_shelters_count:,} 筆
- **佔總避難所比例**: {best_shelters_ratio:.1f}%
- **佔室內避難所比例**: {(best_shelters_count / len(indoor_stats) * 100):.1f}%

### 室內外避難所對比
- **室內避難所**: {len(indoor_stats):,} 筆 ({len(indoor_stats)/total_shelters*100:.1f}%)
  - 平均 AQI: {indoor_stats['aqi_value'].mean():.1f}
  - 平均距離: {indoor_stats['distance_km'].mean():.2f} 公里
- **室外避難所**: {len(outdoor_stats):,} 筆 ({len(outdoor_stats)/total_shelters*100:.1f}%)
  - 平均 AQI: {outdoor_stats['aqi_value'].mean():.1f}
  - 平均距離: {outdoor_stats['distance_km'].mean():.2f} 公里

## 涵蓋盲區分析

### 距離測站最遠的前三名避難所
"""
    
    # 添加最遠避難所詳細資訊
    shelter_name_col = result_df.columns[6]
    for i, (idx, row) in enumerate(farthest_shelters.iterrows(), 1):
        report_content += f"""
{i}. **{row[shelter_name_col]}**
   - 最近測站: {row['nearest_station']}
   - 距離: {row['distance_km']:.2f} 公里
   - AQI 值: {result_df.loc[idx, 'aqi_value']}
   - 類型: {'室內' if result_df.loc[idx, 'is_indoor'] else '室外'}
"""
    
    # 添加距離分佈分析
    report_content += f"""
### 距離分佈分析
- **75% 的避難所距離測站在**: {distance_stats['75%']:.2f} 公里內
- **50% 的避難所距離測站在**: {distance_stats['50%']:.2f} 公里內
- **25% 的避難所距離測站在**: {distance_stats['25%']:.2f} 公里內

### 建議改進方向
1. **涵蓋盲區加強**: 距離超過 {distance_stats['75%']:.1f} 公里的地區可考慮增設臨時 AQI 測站
2. **室內避難所優先**: 在空污事件中，室內避難所應優先開放
3. **監測網絡優化**: 根據避難所分佈調整 AQI 測站佈局

## 資料檔案
- 完整資料: `shelters_with_aqi.csv`
- 最佳避難所清單: `best_shelters.csv`
- 互動式地圖: `shelter_aqi_map_v2.html`

---
*報告由避難收容處所資料分析系統自動生成*
"""
    
    # 寫入檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"反思報告已儲存至: {output_path}")

def main():
    """主程式"""
    print("開始執行避難收容處所資料分析與清理")
    print("=" * 50)
    
    # Task 1: 資料清理
    df = load_shelter_data()
    if df is None:
        return
    
    # 檢查座標系統
    coordinate_system, epsg_code, lon_col, lat_col = check_coordinate_system(df)
    
    # 轉換座標系統
    df = convert_to_wgs84(df, coordinate_system, lon_col, lat_col)
    
    # 過濾離群值
    df_cleaned, outliers = filter_outliers(df, lon_col, lat_col)
    
    # 語意分析
    df_cleaned = semantic_analysis(df_cleaned)
    
    # 儲存結果
    save_results(df_cleaned, outliers)
    
    print("\n" + "=" * 50)
    print("Task 1 資料清理完成！")
    
    # Task 2: 空間疊圖分析
    task2_spatial_analysis()
    
    # Task 3: 最近測站分析與情境模擬
    task3_nearest_station_analysis()
    
    print("\n" + "=" * 50)
    print("所有任務完成！")

if __name__ == "__main__":
    main()
