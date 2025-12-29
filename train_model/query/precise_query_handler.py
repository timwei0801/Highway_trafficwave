"""
精確查詢處理器
使用 Pandas DataFrame 直接查詢 CSV 資料
"""

import os
import re
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from train_model.utils.config_manager import get_config_manager
from train_model.query.intent_classifier import ParsedQuery, QueryIntent


@dataclass
class QueryResult:
    """查詢結果"""
    success: bool
    answer: str
    data: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    confidence: float = 0.0
    error_message: Optional[str] = None


class PreciseQueryHandler:
    """精確查詢處理器 - 使用 DataFrame 直接查詢"""

    def __init__(self, config_path: str = None):
        """初始化處理器"""
        if config_path:
            os.environ['RAG_CONFIG_PATH'] = config_path

        self.config_manager = get_config_manager()
        self.config = self.config_manager.get_config()
        self.data_config = self.config['data_processing']

        # 載入資料
        self.highway1_df: Optional[pd.DataFrame] = None
        self.highway3_df: Optional[pd.DataFrame] = None
        self._load_data()

        # 欄位顯示名稱對應
        self.field_display_names = {
            '車道數': '車道數',
            '車道1寬': '第一車道寬度',
            '車道2寬': '第二車道寬度',
            '車道3寬': '第三車道寬度',
            '曲率半徑': '曲率半徑',
            '縱向坡度': '縱向坡度',
            '橫向坡度': '橫向坡度',
            '路幅寬': '路幅寬',
            '全路幅寬': '全路幅寬',
            '鋪面種類': '鋪面種類',
            '內路肩': '內路肩設置',
            '外路肩': '外路肩設置',
            '內路肩寬': '內路肩寬度',
            '外路肩寬': '外路肩寬度',
            '輔助車道1': '輔助車道1',
            '輔助車道1寬': '輔助車道1寬度',
            '經緯度坐標Lon': '經度',
            '經緯度坐標Lat': '緯度',
        }

        # 單位對應
        self.field_units = {
            '車道數': '個',
            '車道1寬': '公尺',
            '車道2寬': '公尺',
            '車道3寬': '公尺',
            '曲率半徑': '公尺',
            '路幅寬': '公尺',
            '全路幅寬': '公尺',
            '內路肩寬': '公尺',
            '外路肩寬': '公尺',
            '輔助車道1寬': '公尺',
            '經緯度坐標Lon': '度',
            '經緯度坐標Lat': '度',
        }

        logger.info("精確查詢處理器初始化完成")

    def _load_data(self):
        """載入 CSV 資料"""
        base_path = self.config_manager.resolve_path(self.data_config['input_data_path'])

        # 載入國道一號資料
        highway1_path = os.path.join(base_path, self.data_config['highway1_file'])
        if os.path.exists(highway1_path):
            self.highway1_df = pd.read_csv(highway1_path, encoding='utf-8')
            logger.info(f"載入國道一號資料: {len(self.highway1_df)} 筆")
        else:
            logger.warning(f"國道一號資料檔案不存在: {highway1_path}")

        # 載入國道三號資料
        highway3_path = os.path.join(base_path, self.data_config['highway3_file'])
        if os.path.exists(highway3_path):
            self.highway3_df = pd.read_csv(highway3_path, encoding='utf-8')
            logger.info(f"載入國道三號資料: {len(self.highway3_df)} 筆")
        else:
            logger.warning(f"國道三號資料檔案不存在: {highway3_path}")

    def handle_query(self, parsed_query: ParsedQuery) -> QueryResult:
        """
        處理精確查詢

        Args:
            parsed_query: 解析後的查詢

        Returns:
            QueryResult: 查詢結果
        """
        if parsed_query.intent == QueryIntent.PRECISE_LOOKUP:
            return self._handle_precise_lookup(parsed_query)
        elif parsed_query.intent == QueryIntent.COMPARISON:
            return self._handle_comparison(parsed_query)
        elif parsed_query.intent == QueryIntent.ANALYSIS:
            # 分析型查詢：先獲取該路段的完整數據
            return self._handle_analysis_data(parsed_query)
        else:
            return QueryResult(
                success=False,
                answer="此查詢類型不適用於精確查詢處理器",
                error_message="不支援的查詢類型"
            )

    def _handle_analysis_data(self, parsed_query: ParsedQuery) -> QueryResult:
        """獲取分析型查詢所需的完整路段數據"""
        # 選擇對應的資料集
        df = self._get_dataframe(parsed_query.highway)
        if df is None:
            return QueryResult(
                success=False,
                answer=f"找不到{parsed_query.highway}的資料",
                error_message="資料集不存在"
            )

        # 查找里程標記
        row = self._find_row_by_mileage(df, parsed_query.mileage_marker, parsed_query.mileage_km)
        if row is None:
            return QueryResult(
                success=False,
                answer=f"找不到 {parsed_query.mileage_marker} 的資料",
                error_message="里程標記不存在"
            )

        # 提取所有重要欄位
        data = {}
        important_fields = [
            '樁號', '車道數', '路幅寬', '曲率半徑', '縱向坡度', '橫向坡度',
            '鋪面種類', '內路肩', '外路肩', '內路肩寬', '外路肩寬',
            '輔助車道1', '經緯度坐標Lon', '經緯度坐標Lat'
        ]

        for field in important_fields:
            if field in row.index:
                value = row[field]
                if pd.notna(value):
                    data[field] = value

        highway_name = parsed_query.highway.replace('國道', '國道')
        return QueryResult(
            success=True,
            answer=f"已獲取 {highway_name} {parsed_query.mileage_marker} 路段數據",
            data=data,
            source=f"{highway_name} CSV 資料",
            confidence=1.0
        )

    def _handle_precise_lookup(self, parsed_query: ParsedQuery) -> QueryResult:
        """處理精確查詢"""
        # 選擇對應的資料集
        df = self._get_dataframe(parsed_query.highway)
        if df is None:
            return QueryResult(
                success=False,
                answer=f"找不到{parsed_query.highway}的資料",
                error_message="資料集不存在"
            )

        # 查找里程標記
        row = self._find_row_by_mileage(df, parsed_query.mileage_marker, parsed_query.mileage_km)
        if row is None:
            # 嘗試尋找最接近的里程
            closest = self._find_closest_mileage(df, parsed_query.mileage_km)
            if closest is not None:
                suggestion = f"找不到 {parsed_query.mileage_marker}，最接近的是 {closest['樁號']}"
                return QueryResult(
                    success=False,
                    answer=suggestion,
                    error_message="里程標記不存在"
                )
            return QueryResult(
                success=False,
                answer=f"找不到里程 {parsed_query.mileage_marker} 的資料",
                error_message="里程標記不存在"
            )

        # 提取目標欄位
        if parsed_query.target_field:
            return self._extract_single_field(row, parsed_query)
        else:
            return self._extract_all_fields(row, parsed_query)

    def _handle_comparison(self, parsed_query: ParsedQuery) -> QueryResult:
        """處理比較查詢"""
        if not parsed_query.comparison_locations or len(parsed_query.comparison_locations) < 2:
            return QueryResult(
                success=False,
                answer="比較查詢需要至少兩個位置",
                error_message="位置數量不足"
            )

        results = []
        for location in parsed_query.comparison_locations:
            highway = location.get('highway') or parsed_query.highway or '國道1號'
            df = self._get_dataframe(highway)
            if df is None:
                continue

            marker = location.get('marker')
            km = location.get('km')
            row = self._find_row_by_mileage(df, marker, km)

            if row is not None:
                results.append({
                    'location': marker,
                    'highway': highway,
                    'data': row.to_dict()
                })

        if len(results) < 2:
            return QueryResult(
                success=False,
                answer="無法找到足夠的位置資料進行比較",
                error_message="資料不足"
            )

        # 生成比較結果
        return self._generate_comparison_answer(results, parsed_query.target_field)

    def _get_dataframe(self, highway: str) -> Optional[pd.DataFrame]:
        """根據國道名稱取得對應的 DataFrame"""
        if highway is None:
            return self.highway1_df  # 預設使用國道一號

        if '1' in highway or '一' in highway:
            return self.highway1_df
        elif '3' in highway or '三' in highway:
            return self.highway3_df
        else:
            return self.highway1_df

    def _find_row_by_mileage(self, df: pd.DataFrame, marker: str, km: float) -> Optional[pd.Series]:
        """根據里程標記找到對應的資料行"""
        if df is None:
            return None

        # 嘗試精確匹配樁號
        if marker:
            # 標準化樁號格式
            normalized_marker = self._normalize_mileage_marker(marker)
            exact_match = df[df['樁號'] == normalized_marker]
            if not exact_match.empty:
                return exact_match.iloc[0]

            # 嘗試不同格式的匹配
            for _, row in df.iterrows():
                if self._normalize_mileage_marker(str(row['樁號'])) == normalized_marker:
                    return row

        # 嘗試根據里程公尺數匹配
        if km is not None:
            target_meters = int(km * 1000)
            # 容忍50公尺的誤差
            close_match = df[abs(df['里程'] - target_meters) <= 50]
            if not close_match.empty:
                return close_match.iloc[0]

        return None

    def _normalize_mileage_marker(self, marker: str) -> str:
        """標準化里程標記格式"""
        # 移除空格
        marker = marker.strip()

        # 嘗試解析 034K+000 格式
        match = re.match(r'(\d+)K\+(\d+)', marker, re.IGNORECASE)
        if match:
            km = int(match.group(1))
            meters = int(match.group(2))
            return f"{km:03d}K+{meters:03d}"

        # 嘗試解析 34K 格式
        match = re.match(r'(\d+)K$', marker, re.IGNORECASE)
        if match:
            km = int(match.group(1))
            return f"{km:03d}K+000"

        return marker

    def _find_closest_mileage(self, df: pd.DataFrame, target_km: float) -> Optional[pd.Series]:
        """找到最接近的里程"""
        if df is None or target_km is None:
            return None

        target_meters = int(target_km * 1000)
        df_copy = df.copy()
        df_copy['_distance'] = abs(df_copy['里程'] - target_meters)
        closest = df_copy.nsmallest(1, '_distance')

        if not closest.empty:
            return closest.iloc[0]
        return None

    def _extract_single_field(self, row: pd.Series, parsed_query: ParsedQuery) -> QueryResult:
        """提取單一欄位的值"""
        field = parsed_query.target_field

        if field not in row.index:
            # 嘗試找到相似的欄位
            similar = [col for col in row.index if field in col or col in field]
            if similar:
                field = similar[0]
            else:
                return QueryResult(
                    success=False,
                    answer=f"找不到欄位: {field}",
                    error_message="欄位不存在"
                )

        value = row[field]
        display_name = self.field_display_names.get(field, field)
        unit = self.field_units.get(field, '')

        # 處理特殊欄位
        if field in ['經緯度坐標Lon', '經緯度坐標Lat']:
            answer = self._format_coordinate_answer(row, field, parsed_query)
        elif field == '車道數':
            answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處的車道數是 **{int(value)}車道**。"
        elif field in ['內路肩', '外路肩']:
            width_field = f"{field}寬"
            width = row.get(width_field, 0)
            if value == '有' and width > 0:
                answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處**有設置{display_name}**，寬度為 **{width:.3f}公尺**。"
            else:
                answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處**無設置{display_name}**。"
        elif field.startswith('輔助車道'):
            width_field = f"{field}寬"
            width = row.get(width_field, 0)
            if pd.notna(value) and value not in ['無', ''] and width > 0:
                answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處有設置 **{value}**，寬度為 **{width:.3f}公尺**。"
            else:
                answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處**無設置輔助車道**。"
        elif field == '鋪面種類':
            answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處的鋪面種類是 **{value}鋪面**。"
        elif field in ['縱向坡度', '橫向坡度']:
            percentage = float(value) * 100 if pd.notna(value) else 0
            answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處的{display_name}是 **{value}**（約 {percentage:.1f}%）。"
        elif field == '曲率半徑':
            value_float = float(value) if pd.notna(value) else 0
            curve_type = self._classify_curve(value_float)
            answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處的曲率半徑是 **{value_float:.3f}公尺**，屬於{curve_type}。"
        else:
            # 通用格式
            if pd.isna(value):
                answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處的{display_name}資料不存在。"
            elif isinstance(value, (int, float)):
                answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處的{display_name}是 **{value}{unit}**。"
            else:
                answer = f"{parsed_query.highway}{parsed_query.mileage_marker}處的{display_name}是 **{value}**。"

        return QueryResult(
            success=True,
            answer=answer,
            data={field: value, '樁號': row['樁號']},
            source=f"{parsed_query.highway} {row['樁號']}",
            confidence=1.0
        )

    def _format_coordinate_answer(self, row: pd.Series, field: str, parsed_query: ParsedQuery) -> str:
        """格式化座標答案"""
        lon = row.get('經緯度坐標Lon', None)
        lat = row.get('經緯度坐標Lat', None)

        if field == '經緯度坐標Lon':
            if pd.notna(lon):
                return f"{parsed_query.highway}{parsed_query.mileage_marker}處的經度是 **{lon:.7f}**。"
            else:
                return f"{parsed_query.highway}{parsed_query.mileage_marker}處的經度資料不存在。"
        else:
            if pd.notna(lat):
                return f"{parsed_query.highway}{parsed_query.mileage_marker}處的緯度是 **{lat:.7f}**。"
            else:
                return f"{parsed_query.highway}{parsed_query.mileage_marker}處的緯度資料不存在。"

    def _classify_curve(self, radius: float) -> str:
        """分類彎道類型"""
        if radius <= 0 or radius > 50000:
            return "**近似直線路段**"
        elif radius < 500:
            return "**急彎路段**"
        elif radius < 1000:
            return "**中等彎道**"
        elif radius < 2000:
            return "**緩彎路段**"
        else:
            return "**近似直線路段**"

    def _extract_all_fields(self, row: pd.Series, parsed_query: ParsedQuery) -> QueryResult:
        """提取所有重要欄位"""
        highway = parsed_query.highway or '國道1號'
        marker = row['樁號']

        answer_parts = [f"**{highway} {marker} 路段資訊：**\n"]

        # 基本資訊
        answer_parts.append(f"- 車道數：{int(row['車道數'])}車道")
        answer_parts.append(f"- 路幅寬：{row['路幅寬']:.3f}公尺")
        answer_parts.append(f"- 鋪面種類：{row['鋪面種類']}鋪面")

        # 幾何特徵
        if pd.notna(row['曲率半徑']) and row['曲率半徑'] > 0:
            curve_type = self._classify_curve(row['曲率半徑'])
            answer_parts.append(f"- 曲率半徑：{row['曲率半徑']:.3f}公尺（{curve_type}）")

        answer_parts.append(f"- 縱向坡度：{row['縱向坡度']}（{float(row['縱向坡度'])*100:.1f}%）")
        answer_parts.append(f"- 橫向坡度：{row['橫向坡度']}（{float(row['橫向坡度'])*100:.1f}%）")

        # 座標
        if pd.notna(row['經緯度坐標Lon']) and pd.notna(row['經緯度坐標Lat']):
            answer_parts.append(f"- 座標：({row['經緯度坐標Lon']:.6f}, {row['經緯度坐標Lat']:.6f})")

        return QueryResult(
            success=True,
            answer='\n'.join(answer_parts),
            data=row.to_dict(),
            source=f"{highway} {marker}",
            confidence=1.0
        )

    def _generate_comparison_answer(self, results: List[Dict], target_field: Optional[str]) -> QueryResult:
        """生成比較結果"""
        if not results:
            return QueryResult(
                success=False,
                answer="無法生成比較結果",
                error_message="結果為空"
            )

        if target_field:
            # 比較特定欄位
            display_name = self.field_display_names.get(target_field, target_field)
            unit = self.field_units.get(target_field, '')

            answer_parts = [f"**{display_name}比較：**\n"]
            for result in results:
                location = result['location']
                highway = result['highway']
                value = result['data'].get(target_field, 'N/A')

                if isinstance(value, (int, float)) and pd.notna(value):
                    answer_parts.append(f"- {highway} {location}：{value}{unit}")
                else:
                    answer_parts.append(f"- {highway} {location}：{value}")

            # 添加比較結論
            values = [r['data'].get(target_field) for r in results if pd.notna(r['data'].get(target_field))]
            if len(values) >= 2 and all(isinstance(v, (int, float)) for v in values):
                if values[0] == values[1]:
                    answer_parts.append(f"\n**結論**：兩處的{display_name}相同。")
                elif values[0] > values[1]:
                    diff = values[0] - values[1]
                    answer_parts.append(f"\n**結論**：{results[0]['location']}比{results[1]['location']}大 {diff:.3f}{unit}。")
                else:
                    diff = values[1] - values[0]
                    answer_parts.append(f"\n**結論**：{results[1]['location']}比{results[0]['location']}大 {diff:.3f}{unit}。")
        else:
            # 比較所有主要欄位
            answer_parts = ["**路段比較：**\n"]
            main_fields = ['車道數', '路幅寬', '曲率半徑', '縱向坡度', '橫向坡度']

            for field in main_fields:
                display_name = self.field_display_names.get(field, field)
                field_values = []
                for result in results:
                    value = result['data'].get(field, 'N/A')
                    field_values.append(f"{result['location']}: {value}")
                answer_parts.append(f"- {display_name}：{' vs '.join(field_values)}")

        return QueryResult(
            success=True,
            answer='\n'.join(answer_parts),
            data={'comparison': results},
            source="比較查詢",
            confidence=1.0
        )


# 測試用
if __name__ == "__main__":
    from train_model.query.intent_classifier import IntentClassifier

    classifier = IntentClassifier()
    handler = PreciseQueryHandler()

    test_queries = [
        "國道一號34K+000處的車道數是多少？",
        "國道一號34K+000處的曲率半徑是多少？",
        "國道一號34K和80K處的車道數有什麼差異？",
        "國道三號44K+700處的經緯度座標是多少？",
        "國道一號50K+000處有設置減速車道嗎？",
    ]

    for query in test_queries:
        parsed = classifier.classify(query)
        print(f"\n{'='*60}")
        print(f"查詢: {query}")
        print(f"意圖: {parsed.intent.value}")

        if parsed.intent in [QueryIntent.PRECISE_LOOKUP, QueryIntent.COMPARISON]:
            result = handler.handle_query(parsed)
            print(f"結果: {result.answer}")
            print(f"成功: {result.success}")
        else:
            print("需要使用 RAG 處理")
