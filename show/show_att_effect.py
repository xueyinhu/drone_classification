import numpy as np
import matplotlib.pyplot as plt


compare_data = {
    "ECM-Net": [86.85, 0, .0, 0],
    "+SE": [87.18, 4192, 0.23, 5.49e-5],
    "+CBAM": [87.56, 4290, 0.71, 1.655e-4],
    "+ECA": [87.13, 3, 0.28, 9.33e-2],
    "+P2A": [88.25, 308, 1.40, 4.545e-3],
}


fig, ax = plt.subplots(figsize=(8, 6))
x_net_name = ['SE', 'CBAM', 'ECA', 'P2P']
y_net_accu = {
    'ECM-Net': np.array([86.85, 86.85, 86.85, 86.85]),
    'with Attention': np.array([0.23, 0.71, 0.28, 1.40])
}
bottom = np.zeros(len(x_net_name))
for f, accu in y_net_accu.items():
    ax.bar(x_net_name, accu, label=f, bottom=bottom)
    bottom += accu
for i in range(len(x_net_name)):
    plt.text(
        x_net_name[i],
        y_net_accu['ECM-Net'][i] + y_net_accu['with Attention'][i] / 2,
        '+' + '%.2f' % y_net_accu['with Attention'][i] + '%',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=14,
        color='g'
    )

ax.legend(loc="upper left")
plt.ylim((86.5, 88.5))
ax.set_xlabel('attention module', fontsize=14)
ax.set_ylabel('model_acc (%)', fontsize=14)
plt.show()


