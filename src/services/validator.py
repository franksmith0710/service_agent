"""
输入验证和异常处理模块

提供统一的输入验证和异常处理机制
"""

import re
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from src.config.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """验证错误异常"""

    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class AgentError(Exception):
    """Agent 执行错误"""

    def __init__(self, message: str, error_code: str = "AGENT_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ToolError(Exception):
    """工具执行错误"""

    def __init__(
        self, message: str, tool_name: str = "", error_code: str = "TOOL_ERROR"
    ):
        self.message = message
        self.tool_name = tool_name
        self.error_code = error_code
        super().__init__(self.message)


class ErrorCode(Enum):
    """错误码枚举"""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    LLM_ERROR = "LLM_ERROR"
    RAG_ERROR = "RAG_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class ValidationResult:
    """验证结果"""

    is_valid: bool
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode] = None


class InputValidator:
    """输入验证器"""

    MAX_MESSAGE_LENGTH = 2000
    MIN_MESSAGE_LENGTH = 1

    @staticmethod
    def validate_message(message: str) -> ValidationResult:
        """
        验证用户消息

        Args:
            message: 用户输入的消息

        Returns:
            ValidationResult: 验证结果
        """
        if not message:
            return ValidationResult(
                is_valid=False,
                error_message="消息不能为空",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        if len(message.strip()) < InputValidator.MIN_MESSAGE_LENGTH:
            return ValidationResult(
                is_valid=False,
                error_message="消息长度不能少于1个字符",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        if len(message) > InputValidator.MAX_MESSAGE_LENGTH:
            return ValidationResult(
                is_valid=False,
                error_message=f"消息长度不能超过{InputValidator.MAX_MESSAGE_LENGTH}个字符",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        return ValidationResult(is_valid=True)

    @staticmethod
    def validate_session_id(session_id: str) -> ValidationResult:
        """
        验证会话 ID

        Args:
            session_id: 会话 ID

        Returns:
            ValidationResult: 验证结果
        """
        if not session_id:
            return ValidationResult(
                is_valid=False,
                error_message="会话 ID 不能为空",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        if len(session_id) > 100:
            return ValidationResult(
                is_valid=False,
                error_message="会话 ID 长度不能超过100个字符",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
            return ValidationResult(
                is_valid=False,
                error_message="会话 ID 只能包含字母、数字、下划线和连字符",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        return ValidationResult(is_valid=True)

    @staticmethod
    def validate_phone(phone: str) -> ValidationResult:
        """
        验证手机号

        Args:
            phone: 手机号

        Returns:
            ValidationResult: 验证结果
        """
        if not phone:
            return ValidationResult(is_valid=True)

        if not re.match(r"^1[3-9]\d{9}$", phone):
            return ValidationResult(
                is_valid=False,
                error_message="手机号格式不正确",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        return ValidationResult(is_valid=True)

    @staticmethod
    def validate_order_id(order_id: str) -> ValidationResult:
        """
        验证订单号

        Args:
            order_id: 订单号

        Returns:
            ValidationResult: 验证结果
        """
        if not order_id:
            return ValidationResult(is_valid=True)

        if len(order_id) < 6 or len(order_id) > 30:
            return ValidationResult(
                is_valid=False,
                error_message="订单号长度应在6-30个字符之间",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        return ValidationResult(is_valid=True)


class ExceptionHandler:
    """异常处理器"""

    @staticmethod
    def handle_exception(e: Exception, context: str = "") -> str:
        """
        处理异常，返回用户友好的错误消息

        Args:
            e: 异常对象
            context: 上下文信息

        Returns:
            str: 用户友好的错误消息
        """
        logger.error(f"Exception in {context}: {type(e).__name__} - {str(e)}")

        if isinstance(e, ValidationError):
            return f"输入验证失败: {e.message}"

        if isinstance(e, ToolError):
            return f"工具执行失败: {e.message}"

        if isinstance(e, AgentError):
            return f"Agent 执行失败: {e.message}"

        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            return "服务连接失败，请检查网络后重试"

        if "model" in str(e).lower() or "llm" in str(e).lower():
            return "AI 模型服务异常，请稍后重试"

        return "抱歉，系统出现了意外错误，请稍后重试"

    @staticmethod
    def log_exception(e: Exception, context: str, extra_info: dict = None):
        """
        记录异常日志

        Args:
            e: 异常对象
            context: 上下文信息
            extra_info: 额外信息
        """
        extra_str = f", extra: {extra_info}" if extra_info else ""
        logger.error(
            f"[{context}] {type(e).__name__}: {str(e)}{extra_str}", exc_info=True
        )
