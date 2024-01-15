import matplotlib.pyplot as plt


x_net_name = [
    'MobileNetV2',
    'ShuffleNetV2_x0.5',
    'InceptionV3',
    'ResNet34',
    'ShuffleNetV2_x1.0',
    'Xception',
    'ECM-Net',
    'ECM-Net with P2P'
]

y_net_accu = [
    77.19, 81.72, 82.02, 84.89, 85.42, 85.43, 86.85, 88.25
]
y_acc_text = [
    .2, -.7, -.7, -.7, -.7, -.7, -.7, -.7
]

y_net_parm = [
    2264389, 354629, 21813029, 21312773, 4023865, 20871725,
    372197, 373121
]
y_pam_test = [
    "2,264,389", "354,629", "21,813,029", "21,312,773", "4,023,865", "20,871,725",
    "372,197", "373,121"
]
y_pam_text = [
    "2,264,389", "354,629", "21,813,029", "21,312,773", "4,023,865", "20,871,725",
    "372,197", "373,121"
]

y_net_flop = [
    3667.67363, 491.67427, 40740.694014, 44890.21955,
    5964.97507, 56041.287262, 223.0272, 223.033632
]
y_flg_test = [
    3667.67, 491.67, 40740.69, 44890.22,
    5964.98, 56041.29, 223.03, 223.03
]
y_flg_text = [
    2000, 2000, -4000, -4000, -4000, -4000, 2000, 2000
]

fig, ax1 = plt.subplots(figsize=(16, 8))

ax1.plot(
    x_net_name, 
    y_net_accu,
    linewidth=4,
    color='b',
    linestyle='-',
    marker='*',
    markersize=20,
    markeredgecolor='r',
)
for i in range(len(x_net_name)):
    plt.text(
        x_net_name[i], 
        y_net_accu[i] + y_acc_text[i], 
        str(y_net_accu[i]) + "%", 
        horizontalalignment='center',
        fontsize=14,
        color='r'
    )

plt.xticks(fontsize=12, rotation=4, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

ax2 = ax1.twinx()
ax2.plot(
    x_net_name, 
    y_net_flop,
    linewidth=4,
    color='g',
    linestyle='--',
    marker='h',
    markersize=20,
    markeredgecolor='y',
)
for i in range(len(x_net_name)):
    plt.text(
        x_net_name[i], 
        y_net_flop[i] + y_flg_text[i], 
        str(y_flg_test[i]) + "M", 
        horizontalalignment='center',
        fontsize=14,
        color='brown'
    )

# ax1.set_xlabel('model_name', fontsize=18)
ax1.set_ylabel('model_acc (%)', fontsize=18, color='#004fff')
ax2.set_ylabel('model_flogs (M)', fontsize=18, color='#00cf40')

plt.xticks(fontsize=12, rotation=4, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid()
plt.show()

