-- 智能客服 Agent 数据库初始化脚本
-- PostgreSQL 15+

-- 创建表
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    phone VARCHAR(20) UNIQUE NOT NULL,
    membership VARCHAR(50) DEFAULT '普通会员',
    points INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    item_name VARCHAR(200) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pay_method VARCHAR(20),
    shipping_address TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS logistics (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL UNIQUE,
    carrier VARCHAR(50) NOT NULL,
    tracking_number VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    current_location TEXT,
    trace JSONB DEFAULT '[]',
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    series VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    cpu VARCHAR(100),
    gpu VARCHAR(100),
    tgp VARCHAR(20),
    screen_size VARCHAR(20),
    resolution VARCHAR(50),
    refresh_rate VARCHAR(20),
    srgb VARCHAR(20),
    dci_p3 VARCHAR(20),
    delta_e VARCHAR(20),
    ram_max VARCHAR(20),
    ssd_slot VARCHAR(20),
    battery_wh VARCHAR(20),
    weight VARCHAR(20),
    thickness VARCHAR(20),
    has_direct_gpu BOOLEAN DEFAULT false,
    has_water_cooling BOOLEAN DEFAULT false,
    has_rgb BOOLEAN DEFAULT false,
    official_price DECIMAL(10, 2),
    target_user TEXT,
    warranty TEXT,
    keyword TEXT,
    UNIQUE(model)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_user_phone ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logistics_order_id ON logistics(order_id);
CREATE INDEX IF NOT EXISTS idx_products_series ON products(series);
CREATE INDEX IF NOT EXISTS idx_products_model ON products(model);

-- 插入测试数据
INSERT INTO users (user_id, username, phone, membership, points) VALUES
    ('U001', '张三', '13800138000', '黄金会员', 5000),
    ('U002', '李四', '13800138001', '铂金会员', 12000),
    ('U003', '王五', '13800138002', '普通会员', 1000)
ON CONFLICT (phone) DO NOTHING;

INSERT INTO orders (order_id, user_id, status, item_name, quantity, price, total_amount, created_at, pay_method, shipping_address) VALUES
    ('20250413001', 'U001', '已发货', '机械革命 蛟龙16 Pro', 1, 7999.00, 7999.00, '2025-04-13 10:30:00', '支付宝', '北京市朝阳区建国路88号'),
    ('20250413002', 'U001', '待发货', '机械革命 极光Pro', 1, 5999.00, 5999.00, '2025-04-14 15:20:00', '微信支付', '上海市浦东新区世纪大道100号'),
    ('20250414001', 'U002', '已完成', '机械革命 耀世16 Pro', 1, 12999.00, 12999.00, '2025-04-10 09:15:00', '支付宝', '广州市天河区天河路385号'),
    ('20250415001', 'U003', '配送中', '机械革命 无界16 Pro', 1, 4999.00, 4999.00, '2025-04-15 14:00:00', '支付宝', '深圳市南山区科技园路1号')
ON CONFLICT (order_id) DO NOTHING;

INSERT INTO logistics (order_id, carrier, tracking_number, status, current_location, trace) VALUES
    ('20250413001', '顺丰', 'SF123456789', '配送中', '北京市', '[{"time":"2025-04-13 14:00","location":"深圳宝安机场","status":"已发货"},{"time":"2025-04-14 09:30","location":"北京分拨中心","status":"运输中"},{"time":"2025-04-14 18:00","location":"北京朝阳区","status":"派送中"}]'),
    ('20250415001', '中通', 'ZT987654321', '运输中', '广州市', '[{"time":"2025-04-15 16:00","location":"深圳","status":"已发货"},{"time":"2025-04-16 10:00","location":"广州","status":"运输中"}]')
ON CONFLICT (order_id) DO NOTHING;

INSERT INTO products (series, model, cpu, gpu, tgp, screen_size, resolution, refresh_rate, srgb, dci_p3, delta_e, ram_max, ssd_slot, battery_wh, weight, thickness, has_direct_gpu, has_water_cooling, has_rgb, official_price, target_user, warranty, keyword) VALUES
    ('耀世', '耀世16 Pro', 'Intel Core Ultra 9 275HX', 'RTX 5090 16GB', '175W', '16.0英寸', '2.5K(2560×1600)', '300Hz', '100%', '100%', 'ΔE<1', '64GB', '2个', '99Wh', '2.50kg', '20.00mm', true, true, true, 12999, '顶级游戏玩家、专业设计师', '整机2年主要部件3年', '游戏本 旗舰 顶配'),
    ('耀世', '耀世16 Pro Ultra', 'Intel Core Ultra 9 275HX', 'RTX 5080 16GB', '175W', '16.0英寸', '2.5K(2560×1600)', '300Hz', '100%', '100%', 'ΔE<1', '64GB', '2个', '99Wh', '2.50kg', '20.00mm', true, true, true, 9999, '高端游戏玩家', '整机2年主要部件3年', '游戏本 高端'),
    ('蛟龙', '蛟龙16 Pro', 'AMD R9 8945HX', 'RTX 5070 Ti', '140W', '16.0英寸', '2.5K(2560×1600)', '240Hz', '100%', '-', 'ΔE<2', '64GB', '2个', '99Wh', '2.45kg', '20.50mm', true, false, true, 7999, '游戏玩家、学生', '整机2年主要部件3年', '游戏本 AMD性价比'),
    ('蛟龙', '蛟龙16 Pro 2025', 'AMD R9 9945HX', 'RTX 5060', '140W', '16.0英寸', '2.5K(2560×1600)', '240Hz', '100%', '-', 'ΔE<2', '64GB', '2个', '99Wh', '2.45kg', '20.50mm', true, false, true, 6999, '主流游戏玩家', '整机2年主要部件3年', '游戏本 主流'),
    ('极光', '极光Pro', 'Intel Core i7-14700HX', 'RTX 4060', '140W', '15.6英寸', '2.5K(2560×1600)', '165Hz', '100%', '-', 'ΔE<2', '32GB', '2个', '46Wh', '2.20kg', '23.90mm', true, false, true, 5999, '学生、入门游戏', '整机2年主要部件3年', '游戏本 性价比'),
    ('极光', '极光Pro 2025', 'Intel Core i5-14500HX', 'RTX 4060', '140W', '15.6英寸', '2.5K(2560×1600)', '165Hz', '100%', '-', 'ΔE<2', '32GB', '2个', '46Wh', '2.20kg', '23.90mm', true, false, true, 5499, '学生、办公', '整机2年主要部件3年', '游戏本 学生'),
    ('无界', '无界16 Pro', 'Intel Core Ultra 5 125H', '集成显卡', '-', '16.0英寸', '2.5K(2560×1600)', '120Hz', '100%', '-', 'ΔE<2', '32GB', '2个', '70Wh', '1.70kg', '17.00mm', false, false, false, 4999, '商务办公、轻度游戏', '整机2年主要部件3年', '轻薄本 办公'),
    ('无界', '无界16 Pro Max', 'AMD Ryzen AI 7 258V', '集成显卡', '-', '16.0英寸', '2.5K(2560×1600)', '120Hz', '100%', '-', 'ΔE<2', '32GB', '2个', '70Wh', '1.75kg', '18.00mm', false, false, false, 5499, '商务办公、AI应用', '整机2年主要部件3年', '轻薄本 AI')
ON CONFLICT (model) DO NOTHING;

-- 验证数据
SELECT 'Users:' AS info, COUNT(*) AS count FROM users
UNION ALL
SELECT 'Orders:', COUNT(*) FROM orders
UNION ALL
SELECT 'Logistics:', COUNT(*) FROM logistics
UNION ALL
SELECT 'Products:', COUNT(*) FROM products;