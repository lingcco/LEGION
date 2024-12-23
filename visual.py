import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = '/mnt/petrelfs/wensiwei/LEGION/groundingLMM/output/GlamFinetuneOS/events.out.tfevents.1733911186.SH-IDC1-10-140-24-66.71939.0'

ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

tags = ea.Tags()['scalars']

plt.figure(figsize=(10, 6))

# 遍历所有标量标签
for tag in tags:
    if tag == 'train/mask_bce_loss':
        events = ea.Scalars(tag)
        values = [event.value for event in events]
        steps = range(len(values)) 
        # 绘制每个标量标签的折线图
        plt.plot(steps, values, marker='o', linestyle='-', label=tag)

# 添加标题和标签
plt.title('Scalar Values Over Steps')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.legend()

# 显示图表
plt.grid(True)
plt.savefig('visual.png', dpi=300)  # 可以设置文件名和分辨率（dpi）
