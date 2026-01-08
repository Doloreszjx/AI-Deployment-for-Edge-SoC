import onnxruntime as ort
import numpy as np
import time
import json

session_options = ort.SessionOptions()
session_options.enable_profiling = True  # 开启性能分析


session = ort.InferenceSession(
    path_or_bytes='yolov8n.onnx',
    sess_options=session_options,
    providers=['CPUExecutionProvider']
)

session_infos = session.get_inputs()[0]
input_name = session_infos.name
dummy_input = np.random.rand(1,3,640,640).astype(np.float32)

session.run(None, {input_name: dummy_input})

profile_file = session.end_profiling()
print("Profile Saved To:", profile_file)

import json
from collections import defaultdict

with open(profile_file, 'r') as f:
    data = json.load(f)

# 1. 收集所有算子信息
operator_info = []
# total_time: 该算子出现的总时间， count： 出现的次数， times： 每次出现的时间列表
op_type_stats = defaultdict(lambda: {'total_time': 0, 'count': 0, 'times': []})

for event in data:
    if 'cat' in event and event['cat'] == 'Node' and 'args' in event:
        args = event['args']
        if 'op_name' in args:
            op_name = args['op_name']
            node_name = event.get('name', '')
            duration = event.get('dur', 0)
        # 添加到对应算子列表中
        operator_info.append({
            'name': node_name,
            'op_name': op_name,
            'duration': duration
        })

        # 统计算子类型
        op_type_stats[op_name]['total_time'] += duration
        op_type_stats[op_name]['count'] += 1
        op_type_stats[op_name]['times'].append(duration)

# Top5（单次总耗时）
print("\n1. Top 5 耗时算子（按单次执行时间）:")
print("-" * 20)

sorted_operator = sorted(operator_info, key=lambda x: x['duration'], reverse=True)

for i, op in enumerate(sorted_operator[:5]):
    print(f"{i+1}. {op['name']}")
    print(f"   算子类型: {op['op_name']}")
    print(f"   耗时: {op['duration']}μs ({op['duration']/1000:.2f}ms)")
    print()

# 占用时间最多的算子类别
print("\n2. 占用时间最多的算子类别:")
print("-" * 20)

sorted_op_types = sorted(op_type_stats.items(),
                         key=lambda x: x[1]['total_time'],
                         reverse=True)

print(f"{'算子类型':<15} {'总耗时(ms)':<12} {'调用次数':<10} {'平均耗时(μs)':<12} {'占比':<8}")
print("-" * 70)

total_inference_time = sum(stats['total_time'] for stats in op_type_stats.values())

for op_type, stats in sorted_op_types:
    total_ms = stats['total_time'] / 1000
    count = stats['count']
    avg_us = stats['total_time'] / count if count > 0 else 0
    percentage = (stats['total_time'] / total_inference_time * 100) if total_inference_time > 0 else 0

    print(f"{op_type:<15} {total_ms:<12.2f} {count:<10} {avg_us:<12.1f} {percentage:<8.1f}%")

# 4. 小算子分析
print("\n3. 小算子分析（单次执行 < 50μs）:")
print("-" * 70)

# 统计小算子
small_ops_threshold = 50  # 定义小算子阈值（微秒）
small_ops_by_type = defaultdict(int)
total_small_ops = 0

for op_type, stats in op_type_stats.items():
    small_count = sum(1 for t in stats['times'] if t < small_ops_threshold)
    small_ops_by_type[op_type] = small_count
    total_small_ops += small_count

if total_small_ops > 0:
    total_all_ops = sum(stats['count'] for stats in op_type_stats.values())
    small_ops_percentage = total_small_ops / total_all_ops * 100

    print(f"小算子总数: {total_small_ops}/{total_all_ops} ({small_ops_percentage:.1f}%)")
    print("\n小算子类型分布:")

    # 按小算子数量排序
    sorted_small_ops = sorted(small_ops_by_type.items(),
                              key=lambda x: x[1],
                              reverse=True)

    for op_type, count in sorted_small_ops:
        if count > 0:
            total_of_type = op_type_stats[op_type]['count']
            percentage_of_type = count / total_of_type * 100 if total_of_type > 0 else 0
            print(f"  {op_type}: {count}/{total_of_type} ({percentage_of_type:.1f}%是该类型的小算子)")
else:
    print("没有发现小算子（<50μs）")