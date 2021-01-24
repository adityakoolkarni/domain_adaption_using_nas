from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brokenaxes import brokenaxes
import matplotlib


# plt.style.use('bmh')
# fig = plt.figure(figsize=(6,5))#


# plt.plot([0.1, 0.5, 1, 2, 3], [4.26, 4.40, 4.38, 4.54, 4.54], marker='o', color='blue')
# plt.ylim(ymin = 4.2)
# plt.ylim(ymax = 4.6)
# # plt.xlim(xmin = 0.935)
# # plt.xlim(xmax = 0.9655)
# plt.xlabel("Lambda",fontsize=13,fontweight='bold')
# plt.ylabel("Error (%)",fontsize=13,fontweight='bold')
# fig.suptitle('CIFAR-10')
# for a, b in zip([0.1, 0.5, 1, 2, 3], [4.26, 4.40, 4.38, 4.54, 4.54]):
#     plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=12)
#     # plt.text(a, c + 0.0, '%.5f' % c, ha='center', va='bottom', fontsize=7)
# # plt.tight_layout()
# plt.savefig('./plot_cifar10.pdf', dpi=250)
# plt.show()

plt.style.use('bmh')
fig = plt.figure(figsize=(6,5))#


plt.plot([0.1, 0.5, 1, 2, 3], [21.52, 21.62, 21.04, 22.50, 22.02], marker='o', color='blue')
plt.ylim(ymin = 21.00)
plt.ylim(ymax = 22.60)
# plt.xlim(xmin = 0.935)
# plt.xlim(xmax = 0.9655)
plt.xlabel("Lambda",fontsize=13,fontweight='bold')
plt.ylabel("Error (%)",fontsize=13,fontweight='bold')
fig.suptitle('CIFAR-100')
for a, b in zip([0.1, 0.5, 1, 2, 3], [21.52, 21.62, 21.04, 22.50, 22.02]):
    plt.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=12)
    # plt.text(a, c + 0.0, '%.5f' % c, ha='center', va='bottom', fontsize=7)
# plt.tight_layout()
plt.savefig('./plot_cifar100.pdf', dpi=250)
plt.show()


# plt.scatter([0.9504], [0.4993], marker='+', color='red', label='DenseNet-121', s=80)
# plt.scatter([0.9646], [0.4872], marker='*', color='red', label='WRN-28-10', s=80)
# plt.scatter([0.9403], [0.5016], marker='o', color='blue', label='AdaRKNet-62', s=80)
# plt.scatter([0.9596], [0.4855], marker='v', color='blue', label='ResNet-62', s=80)
# plt.scatter([0.9362], [0.4807], marker='s', color='green', label='ResNet-50', s=80)
# plt.scatter([0.9443], [0.4732], marker='d', color='green', label='MobileNetV2', s=80)
# plt.legend()
# plt.ylim(ymin = 0.471)
# plt.ylim(ymax = 0.505)
# plt.xlim(xmin = 0.935)
# plt.xlim(xmax = 0.9655)
# y_major_locator=MultipleLocator(0.01)
# x_major_locator=MultipleLocator(0.01)
# ax=plt.gca()
# ax.yaxis.set_major_locator(y_major_locator)
# ax.xaxis.set_major_locator(x_major_locator)
# # plt.tight_layout()
# plt.xlabel("Natural Accuracy",fontsize=13,fontweight='bold')
# plt.ylabel("Adversarial Robustness (PGD-20)",fontsize=13,fontweight='bold')
# fig.suptitle('Performance Misalignment for Differnet Architectures')
# plt.savefig('./plot_performance.pdf', dpi=250)
# plt.show()