"""產生一個 folium 地圖，顯示環保署即時 AQI 資料 (aqx_p_432 API)。

此腳本會：
1. 檢查並安裝所需套件。
2. 從本地 .env 檔讀取 API 金鑰。
3. 抓取全台測站的即時 AQI 資料。
4. 建置互動式 HTML 地圖，並在每個測站加上標記。

用法：
    python aqi_map.py

產生的地圖預設存放在 outputs 資料夾中的 aqi_map.html。
"""

import os
import sys
import subprocess
import math


# ---------------------------------------------------------------------------
# 內建的套件安裝器（若未安裝則自動安裝）
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    "python-dotenv",
    "requests",
    "folium",
    "pandas",
]


def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


for pkg in REQUIRED_PACKAGES:
    name = pkg.split("==")[0]
    try:
        __import__(name)
    except ImportError:
        print(f"Package '{name}' not found; installing...")
        install(pkg)

# now that dependencies are available, import them normally
from dotenv import load_dotenv
import requests
import folium
import pandas as pd


# ---------------------------------------------------------------------------
# 空間運算輔助函式
# ---------------------------------------------------------------------------


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """計算兩地之間的球面距離（公里）。

    使用 haversine 公式，輸入輸出皆為十進制度數與公里。
    """
    # convert decimal degrees to radians
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    # Earth radius in kilometers
    R = 6371.0
    return R * c


# ---------------------------------------------------------------------------
# 設定區段
# ---------------------------------------------------------------------------

# 從腳本目錄載入 .env
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env"))

API_KEY = os.getenv("MOENV_API_KEY")
if not API_KEY:
    raise RuntimeError("API key not found in .env (MOENV_API_KEY)")

# 環保署開放資料平台的 API 端點
ENDPOINT = "https://data.moenv.gov.tw/api/v2/aqx_p_432"

# ---------------------------------------------------------------------------
# 工具函式區
# ---------------------------------------------------------------------------


def fetch_aqi_data(api_key: str) -> pd.DataFrame:
    """Retrieve AQI data from the MOENV API and return as a DataFrame.

    此函式除了呼叫 API 取得資料外，會把原始的 JSON 回應內容儲存在
    `data/aqi_raw.json`，作為本機端的原始檔案備份。

    API 回傳格式可能是含 "records" 欄位的物件，或直接是一個列表；
    函式會先將之標準化成 list，再轉換成 DataFrame。
    """
    params = {
        "api_key": api_key,
        "limit": 1000,
        "offset": 0,
        "format": "json",
    }
    print("Requesting realtime AQI data from EPA...")
    try:
        resp = requests.get(ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print("Error retrieving AQI data:", exc)
        print("Please check your network connection or API availability.")
        return pd.DataFrame()

    json_payload = resp.json()
    # save raw JSON to data folder for reproducibility / inspection
    raw_dir = os.path.join(base_dir, "data")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "aqi_raw.json")
    try:
        with open(raw_path, "w", encoding="utf-8") as f:
            import json

            json.dump(json_payload, f, ensure_ascii=False, indent=2)
        print(f"Raw API response written to {raw_path}")
    except Exception as e:
        print("Failed to write raw JSON:", e)

    if isinstance(json_payload, dict):
        data = json_payload.get("records", [])
    elif isinstance(json_payload, list):
        data = json_payload
    else:
        print("Unexpected JSON format from API:", type(json_payload))
        data = []

    df = pd.DataFrame(data)
    return df


# ---------------------------------------------------------------------------
# 地圖建立
# ---------------------------------------------------------------------------


def create_map(df: pd.DataFrame, output_file: str = None) -> None:
    """建立一個 folium 地圖顯示 AQI 測站並存成 HTML。"""
    # 去掉沒有座標的列
    df = df.dropna(subset=["latitude", "longitude"])
    # 地圖以台灣為中心
    center_lat = df["latitude"].astype(float).mean()
    center_lon = df["longitude"].astype(float).mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    for _, row in df.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except Exception:
            continue

        # Try multiple field name variations for robustness
        site = row.get("sitename") or row.get("SiteName") or row.get("site_name") or "Unknown"
        county = row.get("county") or row.get("County") or row.get("sitename", "") or ""
        aqi_val = row.get("aqi") or row.get("AQI") or ""
        status = row.get("status") or row.get("Status") or ""

        # 建立彈出視窗 HTML，顯示所有可用資訊
        popup_html = f"<b>\u6e2c\u7ad9: {site}</b><br>"
        if county:
            popup_html += f"\u7e23\u5e02: {county}<br>"
        popup_html += f"AQI: {aqi_val}"
        if status:
            popup_html += f"<br>\u29b6\u72c0: {status}"

        popup = folium.Popup(popup_html, max_width=250)

        # 根據 AQI 值決定標記顏色（三層級分類）
        try:
            aqi_num = pd.to_numeric(aqi_val, errors="coerce")
            if pd.isna(aqi_num):
                marker_color = "gray"
            elif aqi_num <= 50:
                marker_color = "green"  # Good (0-50)
            elif aqi_num <= 100:
                marker_color = "yellow"  # Moderate (51-100)
            else:
                marker_color = "red"  # Poor (101+)
        except Exception:
            marker_color = "gray"

        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color=marker_color,
            fill=True,
            fill_opacity=0.7,
            popup=popup,
        ).add_to(m)

    if output_file is None:
        output_file = os.path.join(base_dir, "outputs", "aqi_map.html")
    m.save(output_file)
    print(f"Map saved to {output_file}")


# ---------------------------------------------------------------------------
# 主要流程
# ---------------------------------------------------------------------------


def main():
    df = fetch_aqi_data(API_KEY)
    if df.empty:
        print("No data retrieved.")
        return
    # 印出欄位名稱與首筆資料供除錯用
    print("\nAvailable columns:", df.columns.tolist())
    if not df.empty:
        print("\nFirst record:")
        print(df.iloc[0].to_dict())
    # 計算每筆資料到台北車站的距離（需有經緯度）
    taipei_lat, taipei_lon = 25.0478, 121.5170
    def compute_dist(row):
        try:
            lat = float(row.get("latitude") or row.get("lat"))
            lon = float(row.get("longitude") or row.get("lon"))
        except Exception:
            return None
        return haversine(taipei_lat, taipei_lon, lat, lon)

    df["distance_km"] = df.apply(compute_dist, axis=1)

    # 確保 outputs 資料夾存在
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "aqi_with_distances.csv")
    # 選擇要輸出的欄位
    export_cols = []
    for col in ["SiteName", "sitename", "site_name"]:
        if col in df.columns:
            export_cols.append(col)
            break
    for col in ["County", "county"]:
        if col in df.columns:
            export_cols.append(col)
            break
    for col in ["AQI", "aqi"]:
        if col in df.columns:
            export_cols.append(col)
            break
    export_cols.append("distance_km")
    try:
        df.to_csv(csv_path, columns=export_cols, index=False)
        print(f"Distance CSV written to {csv_path}")
    except Exception as e:
        print("Failed to write CSV:", e)

    # 將地圖存到 outputs 資料夾
    create_map(df)


if __name__ == "__main__":
    main()
