"""
批量按顺序执行当前目录下所有以 step 为前缀的 Python 脚本。
按文件名中的数字编号排序，编号相同则按文件名字母序执行。
自身 (step0.py) 会被排除。
"""

import os
import re
import subprocess
import sys
import time


def _sort_key(filename):
    """提取 step 后的数字作为主排序键，文件名作为次排序键。"""
    match = re.match(r"step(\d+)", filename)
    num = int(match.group(1)) if match else 0
    return (num, filename)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    self_name = os.path.basename(__file__)

    # 收集所有 step*.py 文件，排除自身
    scripts = [
        f for f in os.listdir(script_dir)
        if f.startswith("step") and f.endswith(".py") and f != self_name
    ]
    scripts.sort(key=_sort_key)

    if not scripts:
        print("未找到任何 step*.py 脚本。")
        return

    total = len(scripts)
    print(f"共发现 {total} 个脚本，将按以下顺序执行：")
    for i, s in enumerate(scripts, 1):
        print(f"  [{i}/{total}] {s}")
    print("-" * 60)

    for i, script in enumerate(scripts, 1):
        script_path = os.path.join(script_dir, script)
        print(f"\n{'=' * 60}")
        print(f"[{i}/{total}] 正在运行: {script}")
        print(f"{'=' * 60}")

        start = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_dir,
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n[错误] {script} 执行失败 (返回码: {result.returncode})，耗时 {elapsed:.1f}s")
            print("后续脚本将不再执行。")
            sys.exit(result.returncode)

        print(f"[完成] {script} 执行成功，耗时 {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"全部 {total} 个脚本执行完毕。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
