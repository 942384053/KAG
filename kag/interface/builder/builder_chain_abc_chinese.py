
# -*- coding: utf-8 -*-
# 版权声明，此代码版权归2023年OpenSPG Authors所有
# Copyright 2023 OpenSPG Authors
#
# 遵循Apache License 2.0许可协议，若要使用此文件，必须遵守该协议
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# 可以从该链接获取Apache License 2.0的完整内容
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据此许可分发的软件按“原样”分发，不附带任何形式的明示或暗示的保证或条件
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.
# 导入asyncio库，用于实现异步编程
import asyncio
# 从typing模块导入List，用于类型提示
from typing import List

# 从concurrent.futures模块导入ThreadPoolExecutor和as_completed
# ThreadPoolExecutor用于创建线程池，as_completed用于获取已完成的线程任务
from concurrent.futures import ThreadPoolExecutor, as_completed
# 从kag.common.registry模块导入Registrable类，可能用于注册和管理组件
from kag.common.registry import Registrable
# 从kag.common.utils模块导入flatten_2d_list函数，用于将二维列表展平为一维列表
from kag.common.utils import flatten_2d_list
# 从knext.builder.builder_chain_abc模块导入BuilderChainABC类，这是一个抽象基类，定义了构建链的基本接口
from knext.builder.builder_chain_abc import BuilderChainABC


# 定义KAGBuilderChain类，继承自BuilderChainABC和Registrable
class KAGBuilderChain(BuilderChainABC, Registrable):
    """
    KAGBuilderChain是一个继承自BuilderChainABC和Registrable基类的类。
    它负责构建和执行由有向无环图（DAG）表示的工作流。
    DAG中的每个节点都是BuilderComponent的实例，并且每个节点的输入会被并行处理。
    """

    # 定义invoke方法，用于同步调用构建链来处理输入文件
    def invoke(self, file_path, max_workers=10, **kwargs):
        """
        调用构建链来处理输入文件。

        参数:
            file_path: 要处理的输入文件的路径。
            max_workers (int, 可选): 要使用的最大线程数。默认为10。
            **kwargs: 额外的关键字参数。

        返回:
            List: 构建链的最终输出。
        """

        # 定义内部函数execute_node，用于并行执行构建链中的单个节点
        def execute_node(node, inputs: List[str]):
            """
            使用并行处理执行构建链中的单个节点。

            参数:
                node: 要执行的节点。
                inputs (List[str]): 节点的输入数据列表。

            返回:
                List: 节点的输出。
            """
            # 获取节点类名的最后一部分，用于进度条显示
            node_name = type(node).__name__.split(".")[-1]
            # 创建一个线程池，最大工作线程数为max_workers
            with ThreadPoolExecutor(max_workers) as inner_executor:
                # 为每个输入创建一个线程任务，提交给线程池执行
                inner_futures = [
                    inner_executor.submit(node.invoke, inp) for inp in inputs
                ]
                # 用于存储节点的输出结果
                result = []
                # 导入tqdm库，用于显示进度条
                from tqdm import tqdm

                # 遍历已完成的线程任务，并显示进度条
                for inner_future in tqdm(
                    as_completed(inner_futures),
                    total=len(inner_futures),
                    desc=f"[{node_name}]",  # 进度条的描述信息
                    position=1,  # 进度条的位置
                    leave=False,  # 任务完成后不保留进度条
                ):
                    # 获取线程任务的结果
                    ret = inner_future.result()
                    # 将结果添加到result列表中
                    result.extend(ret)
                # 返回节点的输出结果
                return result

        # 调用build方法构建构建链，传入文件路径和额外的关键字参数
        chain = self.build(file_path=file_path, **kwargs)
        # 获取构建链中的有向无环图（DAG）
        dag = chain.dag
        # 导入networkx库，用于处理图结构
        import networkx as nx

        # 对DAG中的节点进行拓扑排序，确保节点按依赖顺序执行
        nodes = list(nx.topological_sort(dag))
        # 用于存储每个节点的输出结果
        node_outputs = {}
        # 注释掉的代码，原本可能用于记录已处理的节点名称
        # processed_node_names = []
        # 遍历排序后的节点
        for node in nodes:
            # 注释掉的代码，原本可能用于获取节点类名的最后一部分
            # node_name = type(node).__name__.split(".")[-1]
            # 注释掉的代码，原本可能用于将节点名称添加到已处理节点名称列表中
            # processed_node_names.append(node_name)
            # 获取当前节点的所有前驱节点
            predecessors = list(dag.predecessors(node))
            # 如果当前节点没有前驱节点，说明是起始节点，输入为文件路径
            if len(predecessors) == 0:
                node_input = [file_path]
                # 调用execute_node方法执行节点，并获取输出结果
                node_output = execute_node(node, node_input)
            else:
                # 初始化当前节点的输入列表
                node_input = []
                # 遍历当前节点的所有前驱节点
                for p in predecessors:
                    # 将前驱节点的输出结果添加到当前节点的输入列表中
                    node_input.extend(node_outputs[p])
                # 调用execute_node方法执行节点，并获取输出结果
                node_output = execute_node(node, node_input)
            # 将当前节点的输出结果存储到node_outputs字典中
            node_outputs[node] = node_output
        # 找出DAG中所有出度为0的节点，即最终节点
        output_nodes = [node for node in nodes if dag.out_degree(node) == 0]
        # 用于存储最终输出结果
        final_output = []
        # 遍历最终节点
        for node in output_nodes:
            # 如果最终节点有输出结果
            if node in node_outputs:
                # 将最终节点的输出结果添加到最终输出列表中
                final_output.extend(node_outputs[node])

        # 返回最终输出结果
        return final_output

    # 定义ainvoke方法，用于异步调用构建链来处理输入文件
    async def ainvoke(self, file_path, max_concurrency: int = 100, **kwargs):
        """
        异步调用构建链来处理输入文件。

        参数:
            file_path: 要处理的输入文件的路径。
            **kwargs: 额外的关键字参数。

        返回:
            List: 构建链的最终输出。
        """

        # 定义内部异步函数execute_node，用于异步执行构建链中的单个节点
        async def execute_node(node, inputs: list, semaphore: asyncio.Semaphore):
            """
            异步执行构建链中的单个节点。

            参数:
                node: 要执行的节点。
                inputs (List[str]): 节点的输入数据列表。

            返回:
                List: 节点的输出。
            """

            # 定义内部异步函数ainvoke_with_semaphore，用于在信号量控制下异步调用节点的ainvoke方法
            async def ainvoke_with_semaphore(node, item, semaphore):
                # 异步获取信号量，确保并发数不超过最大并发数
                async with semaphore:
                    # 异步调用节点的ainvoke方法，并返回结果
                    return await node.ainvoke(item)

            # 用于存储异步任务
            tasks = []
            # 遍历节点的输入数据
            for item in inputs:
                # 创建一个异步任务，并添加到tasks列表中
                task = asyncio.create_task(
                    ainvoke_with_semaphore(node, item, semaphore)
                )
                tasks.append(task)
            # 等待所有异步任务完成，并获取结果
            results = await asyncio.gather(*tasks)
            # 将二维结果列表展平为一维列表并返回
            return flatten_2d_list(results)

        # 调用build方法构建构建链，传入文件路径和额外的关键字参数
        chain = self.build(file_path=file_path, **kwargs)
        # 获取构建链中的有向无环图（DAG）
        dag = chain.dag
        # 导入networkx库，用于处理图结构
        import networkx as nx

        # 对DAG中的节点按拓扑层次进行分组
        node_generations = list(nx.topological_generations(dag))
        # 将分组后的节点展平为一维列表
        nodes = flatten_2d_list(node_generations)
        # 用于存储每个节点的输出结果
        node_outputs = {}
        # 创建一个异步信号量，用于控制最大并发数
        semaphore = asyncio.Semaphore(max_concurrency)
        # 遍历节点分组
        for parallel_nodes in node_generations:
            # 用于存储异步任务
            tasks = []
            # 遍历当前分组中的节点
            for node in parallel_nodes:
                # 获取当前节点的所有前驱节点
                predecessors = list(dag.predecessors(node))
                # 从前驱节点收集当前节点的输入数据
                if len(predecessors) == 0:
                    node_input = [file_path]
                    # 创建一个异步任务，并添加到tasks列表中
                    task = asyncio.create_task(
                        execute_node(node, node_input, semaphore)
                    )
                else:
                    # 初始化当前节点的输入列表
                    node_input = []
                    # 遍历当前节点的所有前驱节点
                    for p in predecessors:
                        # 将前驱节点的输出结果添加到当前节点的输入列表中
                        node_input.extend(node_outputs[p])
                    # 创建一个异步任务，并添加到tasks列表中
                    task = asyncio.create_task(
                        execute_node(node, node_input, semaphore)
                    )
                tasks.append(task)
            # 等待当前分组中的所有异步任务完成，并获取结果
            outputs = await asyncio.gather(*tasks)
            # 遍历当前分组中的节点和对应的输出结果
            for node, node_output in zip(parallel_nodes, outputs):
                # 将当前节点的输出结果存储到node_outputs字典中
                node_outputs[node] = node_output

        # 找出DAG中所有出度为0的节点，即最终节点
        output_nodes = [node for node in nodes if dag.out_degree(node) == 0]
        # 用于存储最终输出结果
        final_output = []
        # 遍历最终节点
        for node in output_nodes:
            # 如果最终节点有输出结果
            if node in node_outputs:
                # 将最终节点的输出结果添加到最终输出列表中
                final_output.extend(node_outputs[node])

        # 返回最终输出结果
        return final_output