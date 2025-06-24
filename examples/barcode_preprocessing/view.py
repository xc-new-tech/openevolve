import re, matplotlib.pyplot as plt, pathlib, json

log = pathlib.Path("openevolve_output/logs").glob("*.log").__next__()
success, iteration = [], []
for line in open(log):
    # 匹配迭代记录: "Iteration X: Child ... success_rate=Y"
    m = re.search(r"Iteration (\d+):.*success_rate=([0-9.]+)", line)
    if m:
        iteration.append(int(m.group(1)))
        success.append(float(m.group(2)))

if iteration and success:
    plt.figure(figsize=(10, 6))
    plt.plot(iteration, success, 'o-', linewidth=2, markersize=6)
    plt.xlabel("Iteration")
    plt.ylabel("Success Rate")
    plt.title("OpenEvolve - Barcode Preprocessing Success Rate Evolution")
    plt.grid(True, alpha=0.3)
    plt.show()
    print(f"找到 {len(iteration)} 个数据点")
    print(f"成功率范围: {min(success):.3f} - {max(success):.3f}")
else:
    print("未在日志中找到进化数据")