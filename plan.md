# 智能客服 Agent 改进计划

## 背景问题

当前系统存在数据分布混乱的问题：
- 产品详细参数同时存在于 `data/kb_products.txt` 和 PostgreSQL `products` 表
- 用户咨询产品时检索 Chroma（文本知识库），但产品详情在 PostgreSQL
- 两处数据可能不一致，检索逻辑复杂，响应延迟高

## 改进目标

1. **数据源统一**：产品数据统一存储在 Chroma 向量库，PostgreSQL 只保留订单/物流/用户数据
2. **简化检索逻辑**：产品咨询只检索 Chroma，无需再查 PostgreSQL
3. **统一数据维护**：只更新 `jxgm.md` 一个文件即可

## 改进方案

### 数据层改进

#### 1. 替换知识库数据源

| 现状 | 改进后 |
|------|--------|
| `data/kb_brand.txt` | 删除，合并到 `jxgm.md` |
| `data/kb_products.txt` | 删除，合并到 `jxgm.md` |
| `data/kb_pre_sales.txt` | 删除，合并到 `jxgm.md` |
| `data/kb_after_sales.txt` | 删除，合并到 `jxgm.md` |
| `jxgm.md` | **作为唯一知识库文件** |

#### 2. 清理 PostgreSQL products 表

删除 `products` 表（因为产品数据已统一到 Chroma）

```sql
DROP TABLE IF EXISTS products;
```

### 代码层改进

#### 1. 修改 rag.py

- 更新 `CATEGORY_MAP`，按 `jxgm.md` 章节结构定义分类
- 修改 `load_knowledge_files()` 解析 `jxgm.md` 的逻辑
- 添加从 `jxgm.md` 提取产品参数的函数（用于构建 Document）

#### 2. 修改 intent.py

- `product` 意图只检索 Chroma，无需查询 PostgreSQL

#### 3. 修改 agent.py

- `rag_node` 保持不变，因为现在 Chroma 已包含完整产品参数

### 文件删除清单

```
data/kb_brand.txt      # 删除
data/kb_products.txt    # 删除
data/kb_pre_sales.txt  # 删除
data/kb_after_sales.txt # 删除
```

### 新流程图

```
用户输入 → 意图识别 → 路由
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    chat       product    order/logistics/user/transfer
        │           │              │
        ▼           ▼              ▼
    本地处理     Chroma检索     PostgreSQL查询
        │           │              │
        └───────────┴──────────────┘
                    ▼
              回答生成 + 保存记忆
```

## 实施步骤


### Step 2: 删除 data/kb_*.txt 文件

```bash
rm data/kb_brand.txt
rm data/kb_products.txt
rm data/kb_pre_sales.txt
rm data/kb_after_sales.txt
```

### Step 3: 修改代码

1. **rag.py**
   - 更新 `CATEGORY_MAP` 按 `jxgm.md` 章节结构
   - 重写解析逻辑支持 `jxgm.md` 格式
   - 添加 `init_from_jxgm()` 函数

2. **postgres.py**
   - 删除 `get_product_by_model()` 函数
   - 删除 `get_all_products()` 函数
   - 删除 `get_products_by_series()` 函数
   - 删除 `init_products.sql` 脚本（如有）

### Step 4: 清理 PostgreSQL products 表

```sql
-- 连接数据库后执行
DROP TABLE IF EXISTS products;
```

### Step 5: 重新初始化 Chroma

```bash
python -c "from src.services.rag import init_from_jxgm; init_from_jxgm()"
```

### Step 6: 测试验证

1. 测试产品咨询："蛟龙16 Pro 怎么样" → 验证 Chroma 检索正常
2. 测试订单查询："我的订单" → 验证 PostgreSQL 查询正常
3. 测试知识库问答："保修政策是什么" → 验证 FAQ 检索正常

## 时间线预估

| 步骤 | 工作内容 | 预估时间 |
|------|----------|----------|
| Step 1 | 备份数据 | 5 分钟 |
| Step 2 | 删除旧文件 | 2 分钟 |
| Step 3 | 修改代码 | 30 分钟 |
| Step 4 | 清理数据库 | 5 分钟 |
| Step 5 | 重新初始化 | 10 分钟 |
| Step 6 | 测试验证 | 20 分钟 |
| **合计** | | **约 72 分钟** |

## 风险与回滚

### 风险
- 解析 `jxgm.md` 格式可能有遗漏
- Chroma 重新初始化需要时间

