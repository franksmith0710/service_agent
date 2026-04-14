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
        from src.services.tools import extract_order_id

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
