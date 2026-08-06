"""
测试问题数据集
==================
预定义的测试问题用于评估 RAG 系统质量。
用户可以根据自己的论文集修改这些问题。

每个测试问题包含:
  - question: 问题文本
  - category: 问题类别
  - expected_keywords: 期望回答中包含的关键词（用于辅助评估）
"""

TEST_QUESTIONS = [
    {
        "question": "图像去噪的基本原理是什么？",
        "category": "基础概念",
        "expected_keywords": ["噪声", "去除", "像素", "恢复"],
    },
    {
        "question": "BM3D 算法的核心思想和处理步骤是什么？",
        "category": "经典算法",
        "expected_keywords": ["BM3D", "块匹配", "协同滤波", "3D变换"],
    },
    {
        "question": "深度学习在图像复原中有哪些主要方法？",
        "category": "深度学习",
        "expected_keywords": ["CNN", "GAN", "残差", "注意力"],
    },
    {
        "question": "图像超分辨率重建的常见评价指标有哪些？",
        "category": "评价指标",
        "expected_keywords": ["PSNR", "SSIM", "峰值信噪比"],
    },
    {
        "question": "Transformer 架构在图像复原任务中如何应用？",
        "category": "前沿方法",
        "expected_keywords": ["Transformer", "自注意力", "Swin"],
    },
    {
        "question": "图像去模糊与图像去噪的区别和联系是什么？",
        "category": "对比分析",
        "expected_keywords": ["模糊", "去卷积", "退化模型"],
    },
    {
        "question": "GAN 在图像复原中的优势和挑战是什么？",
        "category": "深度学习",
        "expected_keywords": ["GAN", "对抗", "感知质量", "模式崩溃"],
    },
    {
        "question": "图像复原任务中常用的损失函数有哪些？",
        "category": "训练方法",
        "expected_keywords": ["L1", "L2", "感知损失", "对抗损失"],
    },
]
