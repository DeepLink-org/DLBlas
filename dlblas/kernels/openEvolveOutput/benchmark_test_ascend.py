import os
import importlib.util
import torch
import torch_npu
import time
import logging
import pandas as pd
import traceback
import signal
import gc
import subprocess
import sys

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_errors.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def timeout_handler(signum, frame):
    """超时处理函数"""
    raise TimeoutError("Function execution timed out after 180 seconds")


def reset_npu_state():
    """重置 NPU 状态，确保测试独立性"""
    try:
        torch.npu.empty_cache()
        torch.npu.synchronize()
        # 如果支持，可以添加更多重置操作
        # torch.npu.reset_peak_memory_stats()
        gc.collect()
    except Exception as e:
        logging.warning(f"重置 NPU 状态时出现警告: {e}")


def setup_serial_execution():
    """设置串行执行环境"""
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def benchmark_test_improved(fn, fn_triton, args=(), name="gen_fn", warmup_runs=10, test_runs=100):
    """
    改进的基准测试函数

    Args:
        fn: 原始函数
        fn_triton: Triton优化函数
        args: 函数参数
        name: 测试名称
        warmup_runs: 预热运行次数
        test_runs: 测试运行次数
    """

    print(f"--------------------benchmark_{name} --------------------")

    # 测试前重置状态
    reset_npu_state()

    # Triton版本测试
    triton_times = []
    try:
        fn_triton.to('npu')
        stream = torch.npu.current_stream()

        # 预热阶段
        logging.info(f"开始预热 Triton 版本: {name}")
        stream.synchronize()
        for i in range(warmup_runs):
            fn_triton(*args)
            if i % 5 == 0:  # 每5次同步一次，减少开销
                stream.synchronize()
        stream.synchronize()

        # 正式测试阶段
        logging.info(f"开始测试 Triton 版本: {name}")
        for i in range(test_runs):
            start = time.perf_counter()
            fn_triton(*args)
            stream.synchronize()  # 每次都同步确保准确性
            end = time.perf_counter()
            triton_times.append((end - start) * 1000000)  # 转换为微秒

        # 计算统计信息
        time_compiled_avg = sum(triton_times) / len(triton_times)
        time_compiled_min = min(triton_times)
        time_compiled_max = max(triton_times)
        time_compiled_std = (sum([(t - time_compiled_avg) ** 2 for t in triton_times]) / len(triton_times)) ** 0.5

        print(f"Triton - 平均: {time_compiled_avg:.6f} us, 最小: {time_compiled_min:.6f} us, "
              f"最大: {time_compiled_max:.6f} us, 标准差: {time_compiled_std:.6f} us")

    except Exception as e:
        logging.error(f"Triton 版本运行失败 {name}: {type(e).__name__} - {e}", exc_info=True)
        return None, None, None, f"Triton 运行失败: {str(e)}"

    # 中间清理
    reset_npu_state()

    # 原始版本测试
    eager_times = []
    try:
        fn.to('npu')
        stream = torch.npu.current_stream()

        # 预热阶段
        logging.info(f"开始预热原始版本: {name}")
        stream.synchronize()
        for i in range(warmup_runs):
            fn(*args)
            if i % 5 == 0:
                stream.synchronize()
        stream.synchronize()

        # 正式测试阶段
        logging.info(f"开始测试原始版本: {name}")
        for i in range(test_runs):
            start = time.perf_counter()
            fn(*args)
            stream.synchronize()
            end = time.perf_counter()
            eager_times.append((end - start) * 1000000)

        # 计算统计信息
        time_eager_avg = sum(eager_times) / len(eager_times)
        time_eager_min = min(eager_times)
        time_eager_max = max(eager_times)
        time_eager_std = (sum([(t - time_eager_avg) ** 2 for t in eager_times]) / len(eager_times)) ** 0.5

        print(f"Eager - 平均: {time_eager_avg:.6f} us, 最小: {time_eager_min:.6f} us, "
              f"最大: {time_eager_max:.6f} us, 标准差: {time_eager_std:.6f} us")

        # 计算性能提升
        accelerated = (time_eager_avg - time_compiled_avg) / time_compiled_avg * 100
        print(
            f"{name} 性能提升: {accelerated:.4f}% (Eager: {time_eager_avg:.3f} us, Triton: {time_compiled_avg:.3f} us)")

        return accelerated, time_eager_avg, time_compiled_avg, None

    except Exception as e:
        logging.error(f"原始版本运行失败 {name}: {str(e)}", exc_info=True)
        return None, time_compiled_avg, None, f"Eager 运行失败: {str(e)}"


def load_and_test_operator(folder, output_dir, kernelbench_dir):
    """
    加载并测试单个算子

    Args:
        folder: 算子文件夹名
        output_dir: 输出目录
        kernelbench_dir: 内核基准目录

    Returns:
        list: [算子名称, 原始版本耗时/错误, Triton版本耗时/错误, 性能提升百分比, 错误信息]
    """

    logging.info(f"开始处理算子: {folder}")

    # 解析文件夹名称
    if not (folder.startswith('level') and '_' in folder):
        return [folder, "文件夹名称格式错误", "文件夹名称格式错误", None, "格式错误"]

    parts = folder.split('_')
    if len(parts) < 3 or not parts[0].startswith('level') or not parts[1].isdigit():
        return [folder, "文件夹名称解析失败", "文件夹名称解析失败", None, "解析失败"]

    level = parts[0][5:]  # 去掉 'level' 前缀
    num = parts[1]
    name = '_'.join(parts[2:])

    # 构建文件路径
    kernel_file = os.path.join(kernelbench_dir, f'level{level}', f'{num}_{name}.py')
    best_file = os.path.join(output_dir, folder, 'best', 'best_program.py')

    # 检查文件是否存在
    if not os.path.exists(kernel_file):
        logging.warning(f"内核文件不存在: {kernel_file}")
        return [folder, "内核文件缺失", "内核文件缺失", None, "文件缺失"]

    if not os.path.exists(best_file):
        logging.warning(f"最佳程序文件不存在: {best_file}")
        return [folder, "最佳程序文件缺失", "最佳程序文件缺失", None, "文件缺失"]

    model_old = None
    model_new = None

    try:
        # 重置状态
        reset_npu_state()

        # 加载内核模块
        logging.info(f"加载内核模块: {kernel_file}")
        spec_kernel = importlib.util.spec_from_file_location(f"kernel_{folder}", kernel_file)
        mod_kernel = importlib.util.module_from_spec(spec_kernel)
        spec_kernel.loader.exec_module(mod_kernel)

        if not hasattr(mod_kernel, 'Model'):
            logging.warning(f"内核文件中没有 Model 类: {folder}")
            return [folder, "无 Model 类", "无 Model 类", None, "缺少Model类"]

        Model = mod_kernel.Model

        # 加载最佳程序模块
        logging.info(f"加载最佳程序模块: {best_file}")
        spec_best = importlib.util.spec_from_file_location(f"best_{folder}", best_file)
        mod_best = importlib.util.module_from_spec(spec_best)
        spec_best.loader.exec_module(mod_best)

        # 检查必要的函数和类
        required_attrs = ['ModelNew', 'get_inputs', 'get_init_inputs']
        missing_attrs = [attr for attr in required_attrs if not hasattr(mod_best, attr)]

        if missing_attrs:
            logging.warning(f"最佳程序文件中缺少必要元素 {missing_attrs}: {folder}")
            return [folder, f"缺少{missing_attrs}", f"缺少{missing_attrs}", None, "缺少必要元素"]

        ModelNew = mod_best.ModelNew
        get_inputs = mod_best.get_inputs
        get_init_inputs = mod_best.get_init_inputs

        # 获取初始化参数和输入
        logging.info(f"获取初始化参数: {folder}")
        init_params = get_init_inputs()
        inputs = get_inputs()

        # 实例化模型
        logging.info(f"实例化模型: {folder}")
        model_old = Model(*init_params).to(device='npu')
        model_new = ModelNew(*init_params).to(device='npu')

        # 运行基准测试
        logging.info(f"开始基准测试: {folder}")
        accelerated, time_eager, time_triton, error = benchmark_test_improved(
            model_old, model_new, inputs, folder
        )

        if error:
            return [folder, error, error, None, error]

        if accelerated is not None and time_eager is not None and time_triton is not None:
            return [folder, f"{time_eager:.3f} us", f"{time_triton:.3f} us", f"{accelerated:.4f}%", None]
        else:
            eager_result = f"{time_eager:.3f} us" if isinstance(time_eager, (int, float)) else "测试失败"
            triton_result = f"{time_triton:.3f} us" if isinstance(time_triton, (int, float)) else "测试失败"
            return [folder, eager_result, triton_result, None, "部分测试失败"]

    except Exception as e:
        error_msg = f"未知错误: {str(e)}"
        logging.error(f"处理算子失败 {folder}: {error_msg}\n{traceback.format_exc()}")
        return [folder, error_msg, error_msg, None, traceback.format_exc()]

    finally:
        # 强制清理资源
        try:
            if model_old is not None:
                del model_old
            if model_new is not None:
                del model_new
        except:
            pass

        reset_npu_state()
        logging.info(f"完成算子处理: {folder}")


def filter_operators(output_dir, skip_list=None, include_list=None):
    """
    过滤要测试的算子

    Args:
        output_dir: 输出目录
        skip_list: 要跳过的算子列表
        include_list: 只包含的算子列表（如果指定，只测试这些算子）

    Returns:
        list: 过滤后的算子列表
    """

    if skip_list is None:
        skip_list = ["level3_36_LTSMHn", "level3_37_LTSMCn"]  # 默认跳过的算子

    all_folders = [f for f in os.listdir(output_dir) if f.startswith('level') and '_' in f]

    # 应用包含列表过滤
    if include_list:
        all_folders = [f for f in all_folders if f in include_list]

    # 应用跳过列表过滤
    filtered_folders = [f for f in all_folders if f not in skip_list]

    logging.info(f"总算子数: {len(os.listdir(output_dir))}")
    logging.info(f"符合格式的算子数: {len(all_folders)}")
    logging.info(f"过滤后的算子数: {len(filtered_folders)}")

    return filtered_folders


def run_isolated_benchmark():
    """
    运行隔离的基准测试主函数
    """

    # 配置
    output_dir = './output'
    kernelbench_dir = './kernelbench'

    # 设置串行执行（可选）
    # setup_serial_execution()

    # 过滤算子
    # 选项1: 只测试特定算子
    include_list = ["level1_43_Max_Pooling_3D"]

    # 选项2: 跳过特定算子（注释掉上面的 include_list 来使用）
    skip_list = ["level1_95_CrossEntropyLoss"]
    operators_to_test = filter_operators(output_dir, skip_list=skip_list, include_list=include_list)

    #operators_to_test = filter_operators(output_dir)

    if not operators_to_test:
        logging.error("没有找到要测试的算子")
        return

    logging.info(f"将要测试的算子: {operators_to_test}")

    # 存储结果
    results = []
    successful_tests = 0
    failed_tests = 0

    # 主测试循环
    for i, folder in enumerate(operators_to_test, 1):
        print(f"\n{'=' * 60}")
        print(f"测试进度: {i}/{len(operators_to_test)} - {folder}")
        print(f"{'=' * 60}")

        try:
            # 设置单个测试的超时（可选）
            # signal.signal(signal.SIGALRM, timeout_handler)
            # signal.alarm(300)  # 5分钟超时

            result = load_and_test_operator(folder, output_dir, kernelbench_dir)
            results.append(result)

            # 检查测试是否成功
            if result[4] is None and result[3] is not None:  # 没有错误且有性能数据
                successful_tests += 1
                logging.info(f"✅ 测试成功: {folder} - 性能提升: {result[3]}")
            else:
                failed_tests += 1
                logging.warning(f"❌ 测试失败: {folder} - 错误: {result[4] or '未知错误'}")

            # signal.alarm(0)  # 取消超时

        except KeyboardInterrupt:
            logging.info("用户中断测试")
            break
        except Exception as e:
            failed_tests += 1
            error_msg = f"外层异常: {str(e)}"
            logging.error(f"测试过程中出现异常 {folder}: {error_msg}")
            results.append([folder, error_msg, error_msg, None, traceback.format_exc()])

        # 每次测试后的清理
        reset_npu_state()

        print(f"当前进度: 成功 {successful_tests}, 失败 {failed_tests}")

    # 保存结果
    save_results(results, successful_tests, failed_tests)


def save_results(results, successful_tests, failed_tests):
    """
    保存测试结果到CSV文件

    Args:
        results: 测试结果列表
        successful_tests: 成功测试数
        failed_tests: 失败测试数
    """

    # 创建DataFrame
    df = pd.DataFrame(results, columns=[
        "算子名称",
        "原始版本耗时/错误",
        "Triton版本耗时/错误",
        "性能提升百分比",
        "详细错误信息"
    ])

    # 保存到CSV
    output_file = "benchmark_results_improved.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')

    # 生成统计报告
    print(f"\n{'=' * 60}")
    print("测试完成统计报告")
    print(f"{'=' * 60}")
    print(f"总测试数: {len(results)}")
    print(f"成功测试: {successful_tests}")
    print(f"失败测试: {failed_tests}")
    print(f"成功率: {successful_tests / (successful_tests + failed_tests) * 100:.2f}%")

    # 分析性能提升
    performance_data = []
    for result in results:
        if result[3] and result[3] != "None":
            try:
                perf = float(result[3].replace('%', ''))
                performance_data.append(perf)
            except:
                pass

    if performance_data:
        avg_improvement = sum(performance_data) / len(performance_data)
        max_improvement = max(performance_data)
        min_improvement = min(performance_data)
        print(f"平均性能提升: {avg_improvement:.2f}%")
        print(f"最大性能提升: {max_improvement:.2f}%")
        print(f"最小性能提升: {min_improvement:.2f}%")

    print(f"结果已保存至: {output_file}")

    # 保存详细日志
    logging.info("=" * 60)
    logging.info("测试完成")
    logging.info(f"总测试数: {len(results)}, 成功: {successful_tests}, 失败: {failed_tests}")
    if performance_data:
        logging.info(f"平均性能提升: {avg_improvement:.2f}%")


try:
    logging.info("开始基准测试")
    run_isolated_benchmark()
    logging.info("基准测试完成")
except KeyboardInterrupt:
    logging.info("用户中断程序")
except Exception as e:
    logging.error(f"程序异常退出: {str(e)}\n{traceback.format_exc()}")
finally:
    # 最终清理
    reset_npu_state()
