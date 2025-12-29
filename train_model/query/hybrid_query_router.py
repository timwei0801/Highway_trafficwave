"""
混合查詢路由器
根據查詢意圖自動選擇最佳的查詢策略
"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from train_model.query.intent_classifier import IntentClassifier, ParsedQuery, QueryIntent
from train_model.query.precise_query_handler import PreciseQueryHandler, QueryResult


@dataclass
class HybridQueryResult:
    """混合查詢結果"""
    success: bool
    answer: str
    query_type: str                    # precise / rag / hybrid
    intent: QueryIntent
    confidence: float
    source: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    rag_context: Optional[str] = None  # RAG 檢索到的上下文
    processing_time_ms: Optional[float] = None


class HybridQueryRouter:
    """
    混合查詢路由器

    根據查詢類型自動選擇處理策略：
    1. 精確查詢（里程+欄位）→ DataFrame 直接查詢
    2. 概念解釋 → RAG 向量檢索
    3. 比較分析 → DataFrame 查詢 + LLM 總結
    4. 一般問答 → RAG 向量檢索
    """

    def __init__(self, config_path: str = None):
        """初始化路由器"""
        self.config_path = config_path
        self.intent_classifier = IntentClassifier()
        self.precise_handler = PreciseQueryHandler(config_path)

        # RAG 相關組件（延遲初始化）
        self.rag_chat = None
        self.vector_store = None

        logger.info("混合查詢路由器初始化完成")

    async def initialize_rag(self):
        """初始化 RAG 組件（需要時才初始化）"""
        if self.rag_chat is not None:
            return

        try:
            from train_model.models.ollama_client import OllamaClient, RAGOllamaChat
            from train_model.embeddings.vector_store import VectorStore, RAGRetriever

            ollama_client = OllamaClient(self.config_path)

            if not await ollama_client.check_connection():
                raise Exception("Ollama 服務連接失敗")

            self.vector_store = VectorStore(self.config_path)
            retriever = RAGRetriever(self.vector_store)
            self.rag_chat = RAGOllamaChat(ollama_client, retriever)

            logger.info("RAG 組件初始化完成")
        except Exception as e:
            logger.error(f"RAG 組件初始化失敗: {e}")
            raise

    async def query(self, user_query: str) -> HybridQueryResult:
        """
        處理用戶查詢

        Args:
            user_query: 用戶的原始查詢

        Returns:
            HybridQueryResult: 查詢結果
        """
        import time
        start_time = time.time()

        try:
            # 1. 分類查詢意圖
            parsed_query = self.intent_classifier.classify(user_query)
            logger.info(f"查詢意圖: {parsed_query.intent.value}, 信心度: {parsed_query.confidence:.2f}")

            # 2. 根據意圖選擇處理策略
            if parsed_query.intent == QueryIntent.PRECISE_LOOKUP and parsed_query.confidence >= 0.7:
                result = await self._handle_precise_query(parsed_query)

            elif parsed_query.intent == QueryIntent.CONCEPT_EXPLAIN:
                result = await self._handle_concept_query(parsed_query)

            elif parsed_query.intent == QueryIntent.COMPARISON:
                result = await self._handle_comparison_query(parsed_query)

            elif parsed_query.intent == QueryIntent.RANGE_QUERY:
                result = await self._handle_range_query(parsed_query)

            elif parsed_query.intent == QueryIntent.ANALYSIS:
                result = await self._handle_analysis_query(parsed_query)

            else:
                result = await self._handle_general_query(parsed_query)

            # 3. 計算處理時間
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time

            return result

        except Exception as e:
            logger.error(f"查詢處理失敗: {e}")
            return HybridQueryResult(
                success=False,
                answer=f"查詢處理時發生錯誤：{str(e)}",
                query_type="error",
                intent=QueryIntent.GENERAL_QA,
                confidence=0.0
            )

    async def _handle_precise_query(self, parsed_query: ParsedQuery) -> HybridQueryResult:
        """處理精確查詢"""
        logger.info(f"使用精確查詢處理: {parsed_query.mileage_marker}")

        result = self.precise_handler.handle_query(parsed_query)

        return HybridQueryResult(
            success=result.success,
            answer=result.answer,
            query_type="precise",
            intent=parsed_query.intent,
            confidence=result.confidence,
            source=result.source,
            data=result.data
        )

    async def _handle_concept_query(self, parsed_query: ParsedQuery) -> HybridQueryResult:
        """處理概念解釋查詢"""
        logger.info(f"使用 RAG 處理概念查詢: {parsed_query.concept_term}")

        await self.initialize_rag()

        try:
            # 使用 RAG 獲取概念解釋
            rag_response = await self.rag_chat.chat(parsed_query.original_query)

            return HybridQueryResult(
                success=True,
                answer=rag_response,
                query_type="rag",
                intent=parsed_query.intent,
                confidence=0.85,
                source="概念知識庫"
            )
        except Exception as e:
            logger.error(f"RAG 查詢失敗: {e}")
            return HybridQueryResult(
                success=False,
                answer=f"無法回答此概念問題：{str(e)}",
                query_type="rag",
                intent=parsed_query.intent,
                confidence=0.0
            )

    async def _handle_comparison_query(self, parsed_query: ParsedQuery) -> HybridQueryResult:
        """處理比較查詢"""
        logger.info(f"處理比較查詢: {len(parsed_query.comparison_locations or [])} 個位置")

        # 使用精確查詢處理器處理比較
        result = self.precise_handler.handle_query(parsed_query)

        if result.success:
            return HybridQueryResult(
                success=True,
                answer=result.answer,
                query_type="precise",
                intent=parsed_query.intent,
                confidence=result.confidence,
                source=result.source,
                data=result.data
            )
        else:
            # 如果精確查詢失敗，嘗試使用 RAG
            await self.initialize_rag()
            try:
                rag_response = await self.rag_chat.chat(parsed_query.original_query)
                return HybridQueryResult(
                    success=True,
                    answer=rag_response,
                    query_type="rag",
                    intent=parsed_query.intent,
                    confidence=0.7,
                    source="RAG 檢索"
                )
            except Exception as e:
                return HybridQueryResult(
                    success=False,
                    answer=result.answer,
                    query_type="precise",
                    intent=parsed_query.intent,
                    confidence=0.0
                )

    async def _handle_range_query(self, parsed_query: ParsedQuery) -> HybridQueryResult:
        """處理範圍查詢"""
        logger.info("處理範圍查詢（暫時使用 RAG）")

        # 範圍查詢目前使用 RAG 處理
        # 未來可以實作 DataFrame 聚合計算
        await self.initialize_rag()

        try:
            rag_response = await self.rag_chat.chat(parsed_query.original_query)
            return HybridQueryResult(
                success=True,
                answer=rag_response,
                query_type="rag",
                intent=parsed_query.intent,
                confidence=0.7,
                source="RAG 檢索"
            )
        except Exception as e:
            return HybridQueryResult(
                success=False,
                answer=f"無法處理範圍查詢：{str(e)}",
                query_type="rag",
                intent=parsed_query.intent,
                confidence=0.0
            )

    async def _handle_analysis_query(self, parsed_query: ParsedQuery) -> HybridQueryResult:
        """處理分析型查詢（數據 + LLM 推理）"""
        logger.info(f"處理分析型查詢: {parsed_query.mileage_marker}")

        # 1. 先獲取該路段的精確數據
        precise_result = self.precise_handler.handle_query(parsed_query)

        # 2. 初始化 RAG 組件
        await self.initialize_rag()

        # 3. 組合數據和問題，讓 LLM 進行分析
        if precise_result.success and precise_result.data:
            data_context = self._format_data_for_analysis(precise_result.data, parsed_query)
            analysis_prompt = f"""根據以下道路工程數據，分析用戶的問題。

## 路段數據
{data_context}

## 用戶問題
{parsed_query.original_query}

請根據道路工程數據和專業知識，分析可能的原因並提供建議。"""

            try:
                # 使用 RAG 獲取相關知識，再結合數據進行分析
                from train_model.models.ollama_client import OllamaClient
                from train_model.embeddings.vector_store import RAGRetriever
                ollama_client = OllamaClient(self.config_path)
                retriever = RAGRetriever(self.vector_store)

                # 先檢索相關概念知識
                context_docs = retriever.retrieve_context(parsed_query.original_query, max_context_length=1500)

                full_prompt = f"""你是一位道路工程專家。請根據以下資訊回答用戶的問題。

## 相關專業知識
{context_docs}

## 該路段的實際數據
{data_context}

## 用戶問題
{parsed_query.original_query}

請結合專業知識和實際數據進行分析，提供具體的分析和建議。"""

                response = await ollama_client.generate_response(full_prompt)

                return HybridQueryResult(
                    success=True,
                    answer=response,
                    query_type="analysis",
                    intent=parsed_query.intent,
                    confidence=0.85,
                    source="數據分析 + RAG",
                    data=precise_result.data
                )
            except Exception as e:
                logger.error(f"分析查詢失敗: {e}")
                # 降級：只返回數據
                return HybridQueryResult(
                    success=True,
                    answer=f"該路段數據如下：\n{data_context}\n\n（分析功能暫時無法使用）",
                    query_type="precise",
                    intent=parsed_query.intent,
                    confidence=0.6,
                    data=precise_result.data
                )
        else:
            # 沒有精確數據，使用純 RAG
            return await self._handle_general_query(parsed_query)

    def _format_data_for_analysis(self, data: Dict[str, Any], parsed_query: ParsedQuery) -> str:
        """將數據格式化為分析用的文字描述"""
        lines = []
        lines.append(f"**路段位置**: {parsed_query.highway or '國道1號'} {parsed_query.mileage_marker}")
        if parsed_query.direction:
            lines.append(f"**行駛方向**: {parsed_query.direction}")

        # 關鍵指標
        key_fields = [
            ('車道數', '車道數'),
            ('路幅寬', '路幅寬度'),
            ('曲率半徑', '曲率半徑'),
            ('縱向坡度', '縱向坡度'),
            ('橫向坡度', '橫向坡度'),
            ('鋪面種類', '鋪面種類'),
            ('內路肩', '內路肩'),
            ('外路肩', '外路肩'),
        ]

        for field_name, display_name in key_fields:
            if field_name in data:
                value = data[field_name]
                if field_name == '曲率半徑':
                    if value > 5000:
                        lines.append(f"**{display_name}**: {value}公尺（近似直線）")
                    elif value > 1000:
                        lines.append(f"**{display_name}**: {value}公尺（緩彎）")
                    else:
                        lines.append(f"**{display_name}**: {value}公尺（急彎）")
                elif field_name == '縱向坡度':
                    pct = abs(value * 100)
                    direction = "上坡" if value > 0 else "下坡" if value < 0 else "平坦"
                    lines.append(f"**{display_name}**: {value} ({pct:.1f}% {direction})")
                else:
                    lines.append(f"**{display_name}**: {value}")

        return "\n".join(lines)

    async def _handle_general_query(self, parsed_query: ParsedQuery) -> HybridQueryResult:
        """處理一般問答"""
        logger.info("使用 RAG 處理一般問答")

        await self.initialize_rag()

        try:
            rag_response = await self.rag_chat.chat(parsed_query.original_query)
            return HybridQueryResult(
                success=True,
                answer=rag_response,
                query_type="rag",
                intent=parsed_query.intent,
                confidence=0.7,
                source="RAG 檢索"
            )
        except Exception as e:
            return HybridQueryResult(
                success=False,
                answer=f"無法回答此問題：{str(e)}",
                query_type="rag",
                intent=parsed_query.intent,
                confidence=0.0
            )

    def query_sync(self, user_query: str) -> HybridQueryResult:
        """
        同步版本的查詢方法

        Args:
            user_query: 用戶的原始查詢

        Returns:
            HybridQueryResult: 查詢結果
        """
        return asyncio.run(self.query(user_query))


# 建立全域路由器實例（單例模式）
_router_instance: Optional[HybridQueryRouter] = None


def get_hybrid_router(config_path: str = None) -> HybridQueryRouter:
    """取得混合查詢路由器實例"""
    global _router_instance
    if _router_instance is None:
        _router_instance = HybridQueryRouter(config_path)
    return _router_instance


# 測試用
async def main():
    """測試混合查詢路由器"""
    router = HybridQueryRouter()

    test_queries = [
        # 精確查詢
        "國道一號34K+000處的車道數是多少？",
        "國道一號34K+000處的曲率半徑是多少？",
        "國道三號44K+700處的經緯度座標是多少？",

        # 概念解釋
        "什麼是曲率半徑？",
        "柔性鋪面和剛性鋪面有什麼差別？",

        # 比較查詢
        "國道一號34K和80K處的車道數有什麼差異？",

        # 一般問答
        "國道一號的服務區有哪些？",
    ]

    print("=" * 70)
    print("混合查詢路由器測試")
    print("=" * 70)

    for query in test_queries:
        print(f"\n📝 查詢: {query}")
        print("-" * 50)

        result = await router.query(query)

        print(f"✅ 成功: {result.success}")
        print(f"📊 類型: {result.query_type}")
        print(f"🎯 意圖: {result.intent.value}")
        print(f"📈 信心度: {result.confidence:.2f}")
        print(f"⏱️  處理時間: {result.processing_time_ms:.1f}ms")
        print(f"\n📄 答案:\n{result.answer}")


if __name__ == "__main__":
    asyncio.run(main())
