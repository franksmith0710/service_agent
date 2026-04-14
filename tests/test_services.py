import pytest
from src.config.settings import load_config, AppConfig


class TestConfig:
    """配置模块测试"""

    def test_load_config(self):
        """测试配置加载"""
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.llm_provider in ["ollama", "siliconflow"]
        assert config.llm.model == "qwen3.5:4b"

    def test_llm_config(self):
        """测试 LLM 配置"""
        config = load_config()
        assert config.llm.temperature == 0.7
        assert config.llm.base_url == "http://localhost:11434"


class TestTools:
    """工具模块测试"""

    def test_get_all_tools(self):
        """测试获取所有工具"""
        from src.services.tools import get_all_tools

        tools = get_all_tools()
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "query_order" in tool_names
        assert "query_logistics" in tool_names
        assert "query_user_info" in tool_names
        assert "transfer_to_human" in tool_names

    def test_extract_order_id(self):
        """测试订单号提取"""
        from src.services.intent import extract_order_id

        assert extract_order_id("订单号1234567890") == "1234567890"
        assert extract_order_id("没有订单号") is None


class TestMemory:
    """记忆模块测试"""

    def test_get_memory(self):
        """测试获取记忆实例"""
        from src.services.memory import get_memory

        memory = get_memory("test_session")
        assert memory.session_id == "test_session"
        assert memory.get_messages() == []

    def test_add_message(self):
        """测试添加消息"""
        from src.services.memory import get_memory

        memory = get_memory("test_session_2")
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi there")
        msgs = memory.get_messages()
        assert len(msgs) == 2
        assert msgs[0].content == "Hello"
        assert msgs[1].content == "Hi there"
        memory.clear()


class TestRAG:
    """RAG 模块测试"""

    def test_get_rag(self):
        """测试获取 RAG 实例"""
        from src.services.rag import get_rag

        rag = get_rag()
        assert rag is not None
        docs = rag.similarity_search("订单", k=1)
        assert len(docs) >= 1


class TestDatabase:
    """数据库层测试"""

    def test_get_order_by_id(self):
        """测试按订单号查询"""
        from src.services.postgres import get_order_by_id

        order = get_order_by_id("20250413001")
        assert order is not None
        assert order["order_id"] == "20250413001"
        assert order["status"] == "已发货"

    def test_get_order_by_phone(self):
        """测试按手机号查询订单"""
        from src.services.postgres import get_order_by_phone

        orders = get_order_by_phone("13800138000")
        assert len(orders) >= 1
        assert orders[0]["user_id"] == "U001"

    def test_get_user_by_phone(self):
        """测试按手机号查询用户"""
        from src.services.postgres import get_user_by_phone

        user = get_user_by_phone("13800138000")
        assert user is not None
        assert user["username"] == "张三"
        assert user["membership"] == "黄金会员"

    def test_get_logistics_by_order(self):
        """测试查询物流信息"""
        from src.services.postgres import get_logistics_by_order

        logistics = get_logistics_by_order("20250413001")
        assert logistics is not None
        assert logistics["carrier"] == "顺丰"
        assert logistics["tracking_number"] == "SF123456789"

    def test_search_orders(self):
        """测试订单搜索"""
        from src.services.postgres import search_orders

        results = search_orders("蛟龙")
        assert len(results) >= 1
        assert "蛟龙" in results[0]["items"][0]["name"]


class TestIntent:
    """意图识别测试"""

    def test_recognize_intent_order(self):
        """测试订单查询意图识别"""
        from src.services.intent import recognize_intent, Intent

        intent = recognize_intent("我想查一下订单20250413001")
        assert intent == Intent.ORDER_QUERY

    def test_recognize_intent_logistics(self):
        """测试物流查询意图识别"""
        from src.services.intent import recognize_intent, Intent

        intent = recognize_intent("帮我看看物流到哪了")
        assert intent == Intent.LOGISTICS_QUERY

    def test_recognize_intent_transfer(self):
        """测试转人工意图识别"""
        from src.services.intent import recognize_intent, Intent

        intent = recognize_intent("我要投诉客服")
        assert intent == Intent.TRANSFER_HUMAN

    def test_recognize_intent_chat(self):
        """测试闲聊意图识别"""
        from src.services.intent import recognize_intent, Intent

        intent = recognize_intent("你们这个电脑怎么样")
        assert intent == Intent.GENERAL_CHAT


class TestValidator:
    """验证器模块测试"""

    def test_validate_message_valid(self):
        """测试有效消息验证"""
        from src.services.validator import InputValidator

        result = InputValidator.validate_message("你好，我想查订单")
        assert result.is_valid is True

    def test_validate_message_empty(self):
        """测试空消息验证"""
        from src.services.validator import InputValidator

        result = InputValidator.validate_message("")
        assert result.is_valid is False
        assert "不能为空" in result.error_message

    def test_validate_message_too_long(self):
        """测试过长消息验证"""
        from src.services.validator import InputValidator

        result = InputValidator.validate_message("a" * 3000)
        assert result.is_valid is False
        assert "超过" in result.error_message

    def test_validate_session_id_valid(self):
        """测试有效会话 ID 验证"""
        from src.services.validator import InputValidator

        result = InputValidator.validate_session_id("test_session_123")
        assert result.is_valid is True

    def test_validate_session_id_invalid(self):
        """测试无效会话 ID 验证"""
        from src.services.validator import InputValidator

        result = InputValidator.validate_session_id("test@session")
        assert result.is_valid is False

    def test_validate_phone_valid(self):
        """测试有效手机号验证"""
        from src.services.validator import InputValidator

        result = InputValidator.validate_phone("13800138000")
        assert result.is_valid is True

    def test_validate_phone_invalid(self):
        """测试无效手机号验证"""
        from src.services.validator import InputValidator

        result = InputValidator.validate_phone("12345678901")
        assert result.is_valid is False

    def test_validate_order_id_valid(self):
        """测试有效订单号验证"""
        from src.services.validator import InputValidator

        result = InputValidator.validate_order_id("20250413001")
        assert result.is_valid is True

    def test_validate_order_id_too_short(self):
        """测试过短订单号验证"""
        from src.services.validator import InputValidator

        result = InputValidator.validate_order_id("123")
        assert result.is_valid is False


class TestAPI:
    """API 端点测试"""

    def test_api_import(self):
        """测试 API 模块可导入"""
        from src.services import agent as agent_module
        from src.services import postgres as db_module

        assert agent_module is not None
        assert db_module is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
