# RAG 学术文献智能问答工具 — 简历介绍

## 项目简介（Project Brief）

面向图像复原领域研究者的**私有化本地 RAG 检索问答系统**。基于 LangChain + FAISS 搭建 RAG 全链路（PDF 解析 → 向量化 → 向量检索 + 重排序 → LLM 生成），并采用 **LangGraph** 将流程编排为可观测、可分支的**多智能体协作架构**；同时提供 PySide6 桌面 GUI 与 CLI 双入口，支持流式回答、引用溯源与多轮对话。区别于通用问答，系统通过「仅依据检索文献作答 + [片段N] 引用标注 + 文件名/页码溯源」显著降低幻觉。配套 LLM-as-Judge 评估框架与参数实验框架，用数据驱动检索/生成策略迭代。

技术栈：LangChain · LangGraph · FAISS · DashScope（qwen-turbo / text-embedding-v2 / gte-rerank）· PySide6 · pytest · Docker

## 项目经历（Project Experience）

- 设计并落地基于 RAG 的学术文献问答全链路：使用 pypdf 解析并分块论文，DashScope text-embedding-v2 生成向量，FAISS 构建本地向量库；检索阶段融合向量检索与 gte-rerank 重排序（Top-K → Top-N），保证召回质量与相关性。
- 引入 LangGraph 多智能体编排（QueryAgent / RetrieveAgent / JudgeAgent / GenerateAgent / CiteAgent）：QueryAgent 结合多轮历史改写查询并按问题特征自主选择检索策略（vector / hybrid / multi_query），JudgeAgent 基于相关性评估形成「生成 / 通用知识兜底」条件分支，使每个职责成为可观测、可扩展的独立节点。
- 主导开发 PySide6 桌面端：三栏布局（配置面板 / 问答区 / 引用面板），通过 QThread 将索引构建与问答放入后台线程，实现回答逐 token 流式输出与「引用卡片先行渲染」，保证界面不卡顿。
- 实现引用溯源与多轮对话：回答以 [片段N] 标注来源，右侧面板展示文件名、页码、相关性与原文片段，便于核对；支持基于对话历史的查询改写与上下文注入。
- 搭建 LLM-as-Judge 评估框架（metrics / evaluator），以 Faithfulness、Answer Relevancy、Context Precision、Context Recall 四项指标量化回答忠实度与检索质量，并支持批量测试集回归。
- 构建参数实验框架（experiments），系统化对比切片大小、检索 K、Top-N、温度等超参对效果的影响，用真实指标指导迭代而非凭感觉调参。
- 质量保障与工程化：编写 pytest 单元测试（覆盖配置、文档加载、向量库重排序回退、多智能体图编译与流式契约等）共 58 项全部通过；通过 Dockerfile 容器化部署，降低环境依赖门槛。
