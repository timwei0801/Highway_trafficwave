"""
查詢意圖分類器
根據用戶查詢識別應該使用哪種查詢策略
"""

import re
from typing import Dict, Any, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class QueryIntent(Enum):
    """查詢意圖類型"""
    PRECISE_LOOKUP = "precise_lookup"      # 精確查詢：特定里程的具體數值
    CONCEPT_EXPLAIN = "concept_explain"    # 概念解釋：什麼是曲率半徑
    COMPARISON = "comparison"              # 比較分析：兩個位置的差異
    RANGE_QUERY = "range_query"           # 範圍查詢：某段路的平均值
    ANALYSIS = "analysis"                  # 分析型查詢：結合數據 + LLM 推理（如塞車原因）
    GENERAL_QA = "general_qa"             # 一般問答：需要 RAG 處理


@dataclass
class ParsedQuery:
    """解析後的查詢結構"""
    intent: QueryIntent
    highway: Optional[str] = None         # 國道1號 / 國道3號
    direction: Optional[str] = None       # 南向 / 北向
    mileage_marker: Optional[str] = None  # 034K+000
    mileage_km: Optional[float] = None    # 34.0
    target_field: Optional[str] = None    # 車道數、曲率半徑等
    comparison_locations: Optional[list] = None  # 比較的多個位置
    concept_term: Optional[str] = None    # 要解釋的概念
    original_query: str = ""
    confidence: float = 0.0


class IntentClassifier:
    """意圖分類器"""

    def __init__(self):
        # 里程標記模式：支援多種格式
        self.mileage_patterns = [
            # 標準格式：034K+000, 34K+000
            r'(\d{2,3})K\+(\d{3})',
            # 簡化格式：34K, 50K
            r'(\d{1,3})K(?!\+)',
            # 口語格式：34公里處, 50公里
            r'(\d{1,3})(?:\.?\d*)?(?:公里|km|KM)',
        ]

        # 國道識別模式
        self.highway_patterns = [
            (r'國道[一1]號?', '國道1號'),
            (r'國道[三3]號?', '國道3號'),
            (r'國1', '國道1號'),
            (r'國3', '國道3號'),
            (r'一號?國道', '國道1號'),
            (r'三號?國道', '國道3號'),
        ]

        # 方向識別模式
        self.direction_patterns = [
            (r'[南S][向行]', '南向'),
            (r'[北N][向行]', '北向'),
            (r'往[南高雄屏東]', '南向'),
            (r'往[北基隆台北]', '北向'),
        ]

        # 欄位名稱對應
        self.field_mappings = {
            # 車道相關
            '車道數': '車道數',
            '幾個車道': '車道數',
            '幾車道': '車道數',
            '車道': '車道數',
            '車道1寬': '車道1寬',
            '車道2寬': '車道2寬',
            '車道3寬': '車道3寬',
            '車道寬度': '車道1寬',

            # 坡度相關
            '縱向坡度': '縱向坡度',
            '縱坡': '縱向坡度',
            '橫向坡度': '橫向坡度',
            '橫坡': '橫向坡度',
            '超高': '橫向坡度',

            # 曲率相關
            '曲率半徑': '曲率半徑',
            '曲率': '曲率半徑',
            '半徑': '曲率半徑',
            '彎道': '曲率半徑',

            # 路幅相關
            '路幅寬': '路幅寬',
            '路幅': '路幅寬',
            '全路幅': '全路幅寬',
            '道路寬度': '路幅寬',

            # 鋪面相關
            '鋪面': '鋪面種類',
            '鋪面種類': '鋪面種類',
            '路面': '鋪面種類',

            # 路肩相關
            '內路肩': '內路肩',
            '外路肩': '外路肩',
            '路肩': '外路肩',
            '內路肩寬': '內路肩寬',
            '外路肩寬': '外路肩寬',

            # 輔助車道
            '輔助車道': '輔助車道1',
            '加速車道': '輔助車道1',
            '減速車道': '輔助車道1',

            # 座標相關
            '經度': '經緯度坐標Lon',
            '緯度': '經緯度坐標Lat',
            '座標': '經緯度坐標Lon',
            '坐標': '經緯度坐標Lon',
            '經緯度': '經緯度坐標Lon',
        }

        # 概念關鍵詞
        self.concept_keywords = [
            '什麼是', '是什麼', '定義', '意思', '代表', '表示',
            '解釋', '說明', '介紹', '如何理解',
        ]

        # 比較關鍵詞
        self.comparison_keywords = [
            '比較', '差異', '差別', '不同', '一樣', '相同',
            '和', '與', '跟', 'vs', 'VS',
        ]

        # 分析型關鍵詞（需要數據 + LLM 推理）
        self.analysis_keywords = [
            '為什麼', '原因', '導致', '造成', '因為',
            '塞車', '事故', '危險', '安全',
            '分析', '評估', '建議', '改善',
            '適合', '是否', '能否', '可以嗎',
        ]

        logger.info("意圖分類器初始化完成")

    def classify(self, query: str) -> ParsedQuery:
        """
        分類查詢意圖並解析查詢內容

        Args:
            query: 用戶的原始查詢

        Returns:
            ParsedQuery: 解析後的查詢結構
        """
        query = query.strip()

        # 1. 檢查是否為概念解釋
        if self._is_concept_query(query):
            return self._parse_concept_query(query)

        # 2. 檢查是否為比較查詢
        if self._is_comparison_query(query):
            return self._parse_comparison_query(query)

        # 3. 檢查是否為分析型查詢（有里程 + 分析關鍵詞）
        if self._is_analysis_query(query):
            return self._parse_analysis_query(query)

        # 4. 嘗試解析為精確查詢
        precise_result = self._parse_precise_query(query)
        if precise_result.confidence >= 0.7:
            return precise_result

        # 5. 檢查是否為範圍查詢
        if self._is_range_query(query):
            return self._parse_range_query(query)

        # 6. 預設為一般問答（使用 RAG）
        return ParsedQuery(
            intent=QueryIntent.GENERAL_QA,
            original_query=query,
            confidence=0.5
        )

    def _is_concept_query(self, query: str) -> bool:
        """檢查是否為概念解釋查詢"""
        return any(keyword in query for keyword in self.concept_keywords)

    def _is_analysis_query(self, query: str) -> bool:
        """檢查是否為分析型查詢（需要數據 + LLM 推理）"""
        has_analysis_keyword = any(keyword in query for keyword in self.analysis_keywords)
        has_mileage = self._extract_mileage(query)[0] is not None
        return has_analysis_keyword and has_mileage

    def _parse_analysis_query(self, query: str) -> ParsedQuery:
        """解析分析型查詢"""
        highway = self._extract_highway(query)
        direction = self._extract_direction(query)
        mileage_marker, mileage_km = self._extract_mileage(query)

        # 如果有里程但沒有國道，假設是國道1號
        if mileage_marker and not highway:
            highway = '國道1號'

        return ParsedQuery(
            intent=QueryIntent.ANALYSIS,
            highway=highway,
            direction=direction,
            mileage_marker=mileage_marker,
            mileage_km=mileage_km,
            original_query=query,
            confidence=0.85
        )

    def _is_comparison_query(self, query: str) -> bool:
        """檢查是否為比較查詢"""
        # 包含比較關鍵詞且有多個位置
        has_comparison_keyword = any(keyword in query for keyword in self.comparison_keywords)

        # 檢查是否有多個里程標記
        mileage_matches = []
        for pattern in self.mileage_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            mileage_matches.extend(matches)

        return has_comparison_keyword and len(mileage_matches) >= 2

    def _is_range_query(self, query: str) -> bool:
        """檢查是否為範圍查詢"""
        range_keywords = ['平均', '範圍', '區間', '從.*到', '之間', '整段']
        return any(re.search(keyword, query) for keyword in range_keywords)

    def _parse_concept_query(self, query: str) -> ParsedQuery:
        """解析概念解釋查詢"""
        # 嘗試提取概念術語
        concept_term = None

        # 常見概念列表
        concepts = [
            '曲率半徑', '縱向坡度', '橫向坡度', '柔性鋪面', '剛性鋪面',
            '鋪面', '路肩', '內路肩', '外路肩', '輔助車道', '加速車道',
            '減速車道', '車道寬度', '路幅寬', '超高', '里程標記',
            '交流道', '服務區', '休息站',
        ]

        for concept in concepts:
            if concept in query:
                concept_term = concept
                break

        return ParsedQuery(
            intent=QueryIntent.CONCEPT_EXPLAIN,
            concept_term=concept_term,
            original_query=query,
            confidence=0.9
        )

    def _parse_comparison_query(self, query: str) -> ParsedQuery:
        """解析比較查詢"""
        # 提取所有位置
        locations = []

        # 提取國道
        highway = self._extract_highway(query)

        # 提取所有里程標記
        for pattern in self.mileage_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                mileage_str = match.group()
                mileage_km = self._parse_mileage_to_km(mileage_str)
                if mileage_km is not None:
                    locations.append({
                        'marker': mileage_str,
                        'km': mileage_km,
                        'highway': highway
                    })

        # 提取目標欄位
        target_field = self._extract_target_field(query)

        return ParsedQuery(
            intent=QueryIntent.COMPARISON,
            highway=highway,
            comparison_locations=locations,
            target_field=target_field,
            original_query=query,
            confidence=0.85
        )

    def _parse_precise_query(self, query: str) -> ParsedQuery:
        """解析精確查詢"""
        confidence = 0.0

        # 提取國道
        highway = self._extract_highway(query)
        if highway:
            confidence += 0.3

        # 提取方向
        direction = self._extract_direction(query)

        # 提取里程標記
        mileage_marker, mileage_km = self._extract_mileage(query)
        if mileage_marker:
            confidence += 0.4

        # 提取目標欄位
        target_field = self._extract_target_field(query)
        if target_field:
            confidence += 0.2

        # 如果有里程但沒有國道，假設是國道1號（資料中較多）
        if mileage_marker and not highway:
            highway = '國道1號'
            confidence += 0.1

        return ParsedQuery(
            intent=QueryIntent.PRECISE_LOOKUP,
            highway=highway,
            direction=direction,
            mileage_marker=mileage_marker,
            mileage_km=mileage_km,
            target_field=target_field,
            original_query=query,
            confidence=min(confidence, 1.0)
        )

    def _parse_range_query(self, query: str) -> ParsedQuery:
        """解析範圍查詢"""
        # 簡化處理，未來可以擴展
        return ParsedQuery(
            intent=QueryIntent.RANGE_QUERY,
            highway=self._extract_highway(query),
            target_field=self._extract_target_field(query),
            original_query=query,
            confidence=0.6
        )

    def _extract_highway(self, query: str) -> Optional[str]:
        """提取國道編號"""
        for pattern, highway_name in self.highway_patterns:
            if re.search(pattern, query):
                return highway_name
        return None

    def _extract_direction(self, query: str) -> Optional[str]:
        """提取行駛方向"""
        for pattern, direction in self.direction_patterns:
            if re.search(pattern, query):
                return direction
        return None

    def _extract_mileage(self, query: str) -> Tuple[Optional[str], Optional[float]]:
        """提取里程標記"""
        # 嘗試標準格式：034K+000
        match = re.search(r'(\d{2,3})K\+(\d{3})', query, re.IGNORECASE)
        if match:
            km = int(match.group(1))
            meters = int(match.group(2))
            marker = f"{km:03d}K+{meters:03d}"
            return marker, km + meters / 1000

        # 嘗試簡化格式：34K
        match = re.search(r'(\d{1,3})K(?!\+)', query, re.IGNORECASE)
        if match:
            km = int(match.group(1))
            marker = f"{km:03d}K+000"
            return marker, float(km)

        # 嘗試口語格式：34公里
        match = re.search(r'(\d{1,3})(?:\.(\d+))?(?:公里|km|KM)', query, re.IGNORECASE)
        if match:
            km = int(match.group(1))
            decimal = match.group(2)
            if decimal:
                meters = int(decimal) * (100 if len(decimal) == 1 else 10 if len(decimal) == 2 else 1)
                marker = f"{km:03d}K+{meters:03d}"
                return marker, km + meters / 1000
            else:
                marker = f"{km:03d}K+000"
                return marker, float(km)

        return None, None

    def _parse_mileage_to_km(self, mileage_str: str) -> Optional[float]:
        """將里程標記轉換為公里數"""
        # 嘗試解析 034K+000 格式
        match = re.search(r'(\d+)K\+(\d+)', mileage_str, re.IGNORECASE)
        if match:
            km = int(match.group(1))
            meters = int(match.group(2))
            return km + meters / 1000

        # 嘗試解析簡化格式
        match = re.search(r'(\d+)K', mileage_str, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # 嘗試解析公里格式
        match = re.search(r'(\d+(?:\.\d+)?)', mileage_str)
        if match:
            return float(match.group(1))

        return None

    def _extract_target_field(self, query: str) -> Optional[str]:
        """提取目標欄位"""
        for keyword, field_name in self.field_mappings.items():
            if keyword in query:
                return field_name
        return None


# 測試用
if __name__ == "__main__":
    classifier = IntentClassifier()

    test_queries = [
        "國道一號34K+000處的車道數是多少？",
        "國道一號34K+000處的曲率半徑是多少？",
        "什麼是曲率半徑？",
        "國道一號34K和80K處的車道數有什麼差異？",
        "國道三號44K+700處的經緯度座標是多少？",
        "柔性鋪面和剛性鋪面有什麼差別？",
        "國道一號50公里處的路幅寬是多少？",
        "34K的縱向坡度",
    ]

    for query in test_queries:
        result = classifier.classify(query)
        print(f"\n查詢: {query}")
        print(f"  意圖: {result.intent.value}")
        print(f"  信心度: {result.confidence:.2f}")
        if result.highway:
            print(f"  國道: {result.highway}")
        if result.mileage_marker:
            print(f"  里程: {result.mileage_marker} ({result.mileage_km}km)")
        if result.target_field:
            print(f"  欄位: {result.target_field}")
        if result.concept_term:
            print(f"  概念: {result.concept_term}")
        if result.comparison_locations:
            print(f"  比較位置: {result.comparison_locations}")
