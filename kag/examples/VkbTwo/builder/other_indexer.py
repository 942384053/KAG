# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List  # 移到顶部

# 组件导入
from kag.builder.component.splitter.length_splitter import LengthSplitter
from kag.builder.component.reader.markdown_reader import MarkDownReader as MDReader
from kag.builder.component.extractor.schema_free_extractor import SchemaFreeExtractor
from kag.builder.component.postprocessor.kag_postprocessor import KAGPostProcessor
from kag.builder.component.vectorizer.batch_vectorizer import BatchVectorizer
from kag.builder.component.writer.kg_writer import KGWriter
from kag.builder.model.chunk import Chunk, ChunkTypeEnum
from kag.builder.model.sub_graph import SubGraph  # 补充导入 SubGraph
from kag.common.utils import generate_hash_id
from kag.common.conf import KAG_CONFIG
from kag.interface import VectorizeModelABC

logger = logging.getLogger(__name__)


class ManualKGBuildPipeline:
    def __init__(self, md_file_path):
        self.md_file_path = md_file_path
        config = KAG_CONFIG.all_config["md_kag_builder_pipeline"]
        chain_config = config.get("chain", {})

        splitter_config = chain_config.get("splitter", {})
        self.splitter = LengthSplitter(
            splitter_config['split_length'],
            splitter_config['window_length'],
            splitter_config.get('type', 'sentence')  # 默认值避免 KeyError
        )
        print("Splitter config:", splitter_config)
        print("Splitter instance:", self.splitter.split_length, self.splitter.window_length)
        self.reader = MDReader()
        self.extractor = SchemaFreeExtractor.from_config(chain_config.get("extractor", {}))
        self.post_processor = KAGPostProcessor(**chain_config.get("post_processor", {}).get("params", {}))

        # -------------------------------
        # ✅ 新增：加载并实例化 vectorize_model
        # -------------------------------
        # 新增：加载并实例化 vectorize_model
        vectorize_model_config = KAG_CONFIG.config.get("vectorize_model")
        if not vectorize_model_config:
            raise ValueError("vectorize_model 配置缺失")

        model_type = vectorize_model_config.get("type")

        if model_type == "openai":
            class VectorizeModelMock:
                def __init__(self, config):
                    self.config = config
                    self.api_key = config.get("api_key")
                    self.base_url = config.get("base_url", "").rstrip("/") + "/embeddings"
                    self.model = config.get("model")
                    self.headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }

                def vectorize(self, texts: List[str]) -> List[List[float]]:
                    import requests
                    import json

                    payload = {"model": self.model, "input": texts}
                    response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload))
                    if response.status_code != 200:
                        raise Exception(f"API 调用失败: {response.text}")
                    return [item["embedding"] for item in response.json().get("data", [])]

            vectorize_model_instance = VectorizeModelMock(vectorize_model_config)
        else:
            raise ValueError(f"Unsupported vectorize_model type: {model_type}")
        vectorize_model_config = KAG_CONFIG.config.get("vectorize_model")
        model_type = vectorize_model_config.get("type")
        self.vectorizer = BatchVectorizer(vectorize_model=vectorize_model_instance)
        # -------------------------------

        self.writer = KGWriter()

        # 存储中间结果
        self.raw_text = None
        self.chunks = None
        self.markdown_data = None
        self.triples_list = None
        self.filtered_triples = None
        self.vectorized_results = None

    def read_raw_text(self):
        logger.info(f"Reading Markdown file: {self.md_file_path}")
        with open(self.md_file_path, "r", encoding="utf-8") as f:
            self.raw_text = f.read()
        return self.raw_text

    def split_text(self) -> List[Chunk]:
        if not self.raw_text:
            raise ValueError("Raw text is not loaded. Call read_raw_text first.")
        logger.info("Splitting text into chunks...")

        org_chunk = Chunk(
            id=generate_hash_id("root"),
            name="root_chunk",
            content=self.raw_text,
            type=ChunkTypeEnum.Text
        )

        logger.info(f"Using chunk_size={self.splitter.split_length}, window_length={self.splitter.window_length}")
        print(f"chunk_size: {self.splitter.split_length}, window_length: {self.splitter.window_length}")

        self.chunks = self.splitter.slide_window_chunk(org_chunk)
        logger.info("Text split completed.")

        return self.chunks

    def parse_markdown(self):
        if not self.chunks:
            raise ValueError("Chunks not generated. Call split_text first.")
        logger.info("Parsing markdown content...")
        self.markdown_data = self.reader.read(self.chunks)
        return self.markdown_data

    def extract_triples(self):
        if not self.chunks:
            raise ValueError("Chunks not generated. Call split_text first.")
        logger.info("Extracting triples using LLM...")

        if not hasattr(self, 'extractor') or self.extractor is None:
            config = KAG_CONFIG.all_config["md_kag_builder_pipeline"]
            chain_config = config.get("chain", {})
            self.extractor = SchemaFreeExtractor.from_config(chain_config.get("extractor", {}))

        self.triples_list = []
        for chunk in self.chunks:
            result = self.extractor._invoke(chunk)
            self.triples_list.extend(result)

        return self.triples_list

    def vectorize_entities_relations(self):
        if not self.filtered_triples:
            raise ValueError("Filtered triples not available. Call postprocess_triples first.")
        logger.info("Vectorizing entities and relations...")
        self.vectorized_results = []  # 初始化 vectorized_results
        for graph in self.filtered_triples:
            result = self.vectorizer._generate_embedding_vectors(graph)
            if isinstance(result, list):
                self.vectorized_results.extend(result)
            else:
                self.vectorized_results.append(result)
        return self.vectorized_results

    def postprocess_triples(self):
        if not self.triples_list:
            raise ValueError("Triples not extracted. Call extract_triples first.")
        logger.info("Post-processing triples...")

        processed_results = []
        for triple in self.triples_list:
            result = self.post_processor._invoke(triple)
            processed_results.extend(result)

        self.filtered_triples = processed_results
        return self.filtered_triples

    def write_to_kg(self):
        if not self.vectorized_results:
            raise ValueError("Vectorized results not ready. Call vectorize_entities_relations first.")
        logger.info("Writing to knowledge graph...")

        # ✅ 修改：使用 _invoke 而不是 write()
        for graph in self.vectorized_results:
            self.writer._invoke(graph)

        logger.info("\n\nKnowledge graph built successfully.\n\n")

    # ✅ 新增：支持批量写入
    def write_all(self, graphs: List[SubGraph]):
        results = []
        for graph in graphs:
            result = self.writer._invoke(graph)
            results.extend(result)
        return results

    def run(self):
        """完整流程执行"""
        self.read_raw_text()
        self.split_text()
        self.parse_markdown()
        self.extract_triples()
        self.postprocess_triples()
        self.vectorize_entities_relations()
        self.write_to_kg()


if __name__ == "__main__":
    from pathlib import Path

    current_dir = Path(__file__).parent
    file_path = str(current_dir / "data/sql9.md")

    pipeline = ManualKGBuildPipeline(file_path)


    # 定义通用的 SubGraph 打印函数
    def print_subgraph(idx, subgraph, show_vectors=True):
        print(f"\n{'=' * 40} SubGraph {idx + 1} {'=' * 40}")
        print(f"Nodes ({len(subgraph.nodes)}):")
        for node in subgraph.nodes:
            print(f"  - ID: {node.id}, Properties: {node.properties}")

        print(f"Edges ({len(subgraph.edges)}):")
        for edge in subgraph.edges:
            print(f"  - From: {edge.from_id} → To: {edge.to_id}, Properties: {edge.properties}")

        if show_vectors:
            print("Embedding Vectors:")
            # 这里假设 Node 和 Edge 对象可能有向量属性
            for node in subgraph.nodes:
                if hasattr(node, 'vector') and node.vector is not None:
                    print(f"  Node {node.id}: Vector dimension = {len(node.vector)}")
            for edge in subgraph.edges:
                if hasattr(edge, 'vector') and edge.vector is not None:
                    print(f"  Edge {edge.from_id}→{edge.to_id}: Vector dimension = {len(edge.vector)}")


    # 执行流程
    print(f"\n{'=' * 60}")
    print(f"开始构建知识图谱: {file_path}")
    print(f"{'=' * 60}\n")

    # 读取原始文本
    print("1. 读取原始文本...")
    raw_text = pipeline.read_raw_text()
    print(f"✅ 原始文本长度: {len(raw_text)} 字符")

    # 文本分割
    print("\n2. 分割文本...")
    chunks = pipeline.split_text()
    print(f"✅ 分割后的 chunk 数量: {len(chunks)}")

    # 提取三元组
    print("\n3. 提取知识三元组...")
    triples = pipeline.extract_triples()
    print(f"✅ 提取出的 SubGraph 数量: {len(triples)}")
    for i, subgraph in enumerate(triples[:3]):  # 只展示前3个，避免过多输出
        print_subgraph(i, subgraph, show_vectors=False)
    if len(triples) > 3:
        print(f"... 还有 {len(triples) - 3} 个 SubGraph 未展示")

    # 后处理三元组
    print("\n4. 后处理三元组...")
    filtered_triples = pipeline.postprocess_triples()
    print(f"✅ 处理后的 SubGraph 数量: {len(filtered_triples)}")
    for i, subgraph in enumerate(filtered_triples[:3]):  # 只展示前3个
        print_subgraph(i, subgraph, show_vectors=False)
    if len(filtered_triples) > 3:
        print(f"... 还有 {len(filtered_triples) - 3} 个 SubGraph 未展示")

    # 向量化实体和关系
    print("\n5. 向量化实体和关系...")
    vectorized_results = pipeline.vectorize_entities_relations()
    print(f"✅ 向量化后的结果数量: {len(vectorized_results)}")
    for i, result in enumerate(vectorized_results[:3]):  # 只展示前3个
        print_subgraph(i, result, show_vectors=True)
    if len(vectorized_results) > 3:
        print(f"... 还有 {len(vectorized_results) - 3} 个结果未展示")

    # 写入知识图谱
    print("\n6. 写入知识图谱...")
    pipeline.write_to_kg()
    print(f"\n{'=' * 60}")
    print(f"知识图谱构建完成!")
    print(f"{'=' * 60}")
