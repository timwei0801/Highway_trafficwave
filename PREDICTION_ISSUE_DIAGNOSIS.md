# 預測系統「無足夠資料」問題診斷

## 錯誤訊息
```
2025-10-07 16:08:32,766 - MTSTNetPredictor - WARNING - ⚠️ 無足夠資料進行預測
INFO: 127.0.0.1:63141 - "GET /api/prediction/traffic HTTP/1.1" 503 Service Unavailable
```

## 問題根源

### 1. 錯誤發生位置
**檔案**: `src/models/mt_stnet/realtime_predictor.py`
**函數**: `preprocess_data_for_prediction()`
**行數**: Line 339

```python
if not station_sequences:
    self.logger.warning("⚠️ 無足夠資料進行預測")
    return None
```

### 2. 執行流程

```
API 請求: GET /api/prediction/traffic
    ↓
prediction.py:line 56 - predictor.run_single_prediction()
    ↓
realtime_predictor.py:line 554 - get_realtime_data()
    ↓
取得即時資料 (從 OptimizedIntegratedDataCollectionSystem)
    ↓
realtime_predictor.py:line 559 - preprocess_data_for_prediction()
    ↓
處理每個站點資料:
  - 需要每個站點 >= 6 筆資料 (min_data_points)
  - 需要 12 個時間步長 (input_length)
    ↓
檢查 station_sequences 是否為空
    ↓
如果為空 → 返回 None → API 返回 503 錯誤
```

### 3. 問題原因

#### 原因 A: 資料收集器未初始化
```python
# realtime_predictor.py:line 272-274
if self.data_collector is None:
    if not self.initialize_data_collector():
        return pd.DataFrame()  # 返回空 DataFrame
```

**症狀**:
- 資料收集器初始化失敗
- 沒有載入歷史資料
- `get_cached_data_for_output()` 返回空資料

#### 原因 B: 即時資料不足
```python
# realtime_predictor.py:line 313-336
for station in self.target_stations:
    station_data = data[data['station'] == station].copy()

    if len(station_data) >= self.min_data_points:  # 需要 >= 6 筆
        # 處理資料...
    # 如果資料不足,跳過該站點
```

**症狀**:
- 每個站點的資料點 < 6 筆
- 所有站點都被跳過
- `station_sequences` 為空字典

#### 原因 C: 缺少訓練資料 (train.csv)
```python
# realtime_predictor.py:line 236-239
train_file = self.data_dir / 'Taiwan' / 'train.csv'
if train_file.exists():
    # 計算正規化參數
else:
    # 使用預設值
```

**影響**:
- 正規化參數不準確
- 不影響「無足夠資料」錯誤
- 但影響預測品質

## 診斷步驟

### 步驟 1: 檢查資料目錄
```bash
ls -la data/Taiwan/
```

**期望結果**:
- ✅ Etag.csv 存在
- ✅ 國道一號_整合資料.csv 存在
- ✅ 國道三號_整合資料.csv 存在
- ⚠️ train.csv 不存在 (可選)

### 步驟 2: 檢查即時資料
```bash
ls -la data/realtime_data/ | head -10
```

**期望結果**:
- ✅ 有近期的 detailed_cached_data_*.csv 檔案
- ✅ 檔案時間戳接近當前時間

**實際結果**:
```
✅ 有即時資料檔案
最新: detailed_cached_data_20251007_1609.csv
```

### 步驟 3: 檢查資料收集器
```python
predictor = MTSTNetRealtimePredictor()
predictor.initialize_data_collector()

# 檢查是否成功
if predictor.data_collector is None:
    print("❌ 資料收集器初始化失敗")
else:
    print("✅ 資料收集器初始化成功")
```

### 步驟 4: 檢查資料內容
```python
realtime_data = predictor.get_realtime_data()
print(f"資料筆數: {len(realtime_data)}")
print(f"站點數量: {realtime_data['station'].nunique()}")
print(f"時間範圍: {realtime_data['timestamp'].min()} ~ {realtime_data['timestamp'].max()}")
```

## 解決方案

### 方案 1: 確保資料收集器啟動 (推薦)

**問題**: 資料收集器需要在 API 啟動前初始化

**解決步驟**:

1. 在 FastAPI 啟動時初始化預測器
   ```python
   # main.py 或 api/main.py
   from api.routes.prediction import get_predictor

   @app.on_event("startup")
   async def startup_event():
       predictor = get_predictor()
       predictor.initialize_data_collector()
       print("✅ 預測器初始化完成")
   ```

2. 確保資料收集器載入歷史資料
   ```python
   predictor.data_collector.load_initial_historical_data()
   ```

### 方案 2: 降低資料需求門檻

**問題**: 每個站點需要 6 筆資料太嚴格

**解決步驟**:

修改 `realtime_predictor.py:line 65`:
```python
# 原本
self.min_data_points = 6

# 修改為
self.min_data_points = 3  # 降低到 3 筆
```

### 方案 3: 使用模擬資料 (開發測試用)

**問題**: 沒有足夠的歷史資料進行預測

**解決步驟**:

創建模擬資料生成器:
```python
# 在 preprocess_data_for_prediction() 中
if data.empty or len(data) < self.min_data_points:
    # 使用模擬資料
    data = self._generate_mock_data()
```

### 方案 4: 修改預測 API 錯誤處理

**問題**: 沒有資料時應該提供更友善的錯誤訊息

**解決步驟**:

修改 `prediction.py:line 76-79`:
```python
else:
    # 提供更詳細的錯誤訊息
    error_msg = prediction_result.get('error', '預測系統暫時無法使用')

    # 檢查具體原因
    if '無可用的即時資料' in error_msg:
        detail = "資料收集系統尚未準備就緒,請稍後再試"
    elif '資料預處理失敗' in error_msg:
        detail = "即時資料不足,需要更多歷史資料點"
    else:
        detail = error_msg

    raise HTTPException(status_code=503, detail=f"MT-STNet預測暫時無法使用: {detail}")
```

## 快速修復腳本

創建 `scripts/init_prediction_system.py`:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mt_stnet.realtime_predictor import MTSTNetRealtimePredictor

def initialize_prediction_system():
    print("🚀 初始化預測系統...")

    predictor = MTSTNetRealtimePredictor()

    # 1. 載入模型
    print("📥 載入模型...")
    predictor.load_model()

    # 2. 初始化資料收集器
    print("🔧 初始化資料收集器...")
    if predictor.initialize_data_collector():
        print("✅ 資料收集器初始化成功")

        # 3. 載入歷史資料
        print("📊 載入歷史資料...")
        predictor.data_collector.load_initial_historical_data()

        # 4. 測試單次預測
        print("🔮 測試預測功能...")
        result = predictor.run_single_prediction()

        if result.get('predictions'):
            print(f"✅ 預測成功: {len(result['predictions'])} 個站點")
            print("\n預測結果預覽:")
            for pred in result['predictions'][:3]:
                print(f"  {pred['location_name']}: "
                      f"流量={pred['predicted_flow']:.1f}, "
                      f"速度={pred['predicted_speed']:.1f}km/h, "
                      f"信心度={pred['confidence']:.2f}")
        else:
            print(f"❌ 預測失敗: {result.get('error', '未知錯誤')}")
            print("\n診斷資訊:")
            print(f"  - 資料收集器狀態: {'正常' if predictor.data_collector else '未初始化'}")
            print(f"  - 目標站點數: {len(predictor.target_stations)}")

            # 檢查資料
            realtime_data = predictor.get_realtime_data()
            print(f"  - 即時資料筆數: {len(realtime_data)}")
            if not realtime_data.empty:
                print(f"  - 站點數: {realtime_data['station'].nunique()}")
                print(f"  - 時間範圍: {realtime_data['timestamp'].min()} ~ {realtime_data['timestamp'].max()}")
    else:
        print("❌ 資料收集器初始化失敗")

    print("\n🎯 初始化完成")
    return predictor

if __name__ == "__main__":
    initialize_prediction_system()
```

**執行**:
```bash
python scripts/init_prediction_system.py
```

## 推薦的解決流程

### 立即執行 (2 分鐘)
1. 檢查資料目錄和即時資料 ✅ (已確認有資料)
2. 降低 `min_data_points` 從 6 改為 3
3. 重啟 API 服務

### 短期修復 (10 分鐘)
1. 創建並執行初始化腳本
2. 在 API startup 事件中初始化預測器
3. 改善錯誤訊息

### 長期改善 (1 小時)
1. 實作資料收集器健康檢查
2. 實作資料品質監控
3. 實作降級預測邏輯(資料不足時使用簡化模型)

## 驗證修復

修復後,執行以下測試:

```bash
# 1. 檢查預測 API
curl http://localhost:8000/api/prediction/traffic

# 2. 檢查模型狀態
curl http://localhost:8000/api/prediction/model/status

# 3. 查看日誌
tail -f data/logs/mt_stnet_predictor_*.log
```

**期望結果**:
- ✅ 返回預測資料 (不是 503 錯誤)
- ✅ model_status 顯示 `is_running: true`
- ✅ 日誌顯示成功預測訊息

---

**診斷完成時間**: 2025-10-07
**預計修復時間**: 2-10 分鐘
**嚴重程度**: 中等 (功能無法使用,但有解決方案)
