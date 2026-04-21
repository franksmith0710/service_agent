#!/usr/bin/env python3
"""一键初始化脚本"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 50)
    print("智能客服系统初始化")
    print("=" * 50)

    print("\n[1/2] 初始化知识库...")
    try:
        from src.services.rag import init_from_files

        count = init_from_files()
        print(f"   已导入 {count} 个文档")
    except Exception as e:
        print(f"   知识库初始化失败: {e}")

    print("\n[2/2] 测试数据库连接...")
    try:
        from src.services.postgres import test_connection

        if test_connection():
            print("   数据库连接成功")
        else:
            print("   数据库连接失败")
    except Exception as e:
        print(f"   数据库连接失败: {e}")

    print("\n" + "=" * 50)
    print("初始化完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
