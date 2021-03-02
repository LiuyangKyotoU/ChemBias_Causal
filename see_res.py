import torch
import matplotlib.pyplot as plt
import numpy as np

# QM9
trial = [i for i in range(10)]
bias = ['000', '100', '010', '001', '110', '101', '011', '111']
alpha = [0.0, 100.0]
beta = [0.0, 0.1]

ans = np.zeros((len(trial), len(bias), len(alpha), len(beta), 12))
for i in range(10):
    for j in range(8):
        for m in range(2):
            for n in range(2):
                path = 'trail({})-bias({})-alpha({})-beta({})'.format(
                    trial[i], bias[j], alpha[m], beta[n]
                )
                ans[i][j][m][n] = torch.load('results/' + path + '_mae.pt').numpy()

tmp = np.zeros((10, 8, 12))
for i in range(10):
    for j in range(8):
        path = 'twostep-trail({})-bias({})'.format(trial[i], bias[j])
        tmp[i][j] = torch.load('results/' + path + '_mae.pt').numpy()

targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
std = np.array([1.4974e+00, 8.1632e+00, 5.9892e-01, 1.2743e+00, 1.2858e+00, 2.8021e+02,
                9.0042e-01, 1.0846e+03, 1.0846e+03, 1.0846e+03, 1.0846e+03, 4.0549e+00])

linewidth = 1
fig = plt.figure(figsize=(12, 9))
r = np.linspace(0, 2, 9)
theta = np.pi * r
for y in range(12):
    ax = plt.subplot(3, 4, y + 1, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticklabels(bias)
    ax.set_title('Target ' + targets[y], loc='left')
    ax.set_rlabel_position(120)
    ax.plot(theta, np.append(ans[:, :, 0, 0, y].mean(axis=0), ans[:, 0, 0, 0, y].mean(axis=0)) / std[y],
            label='Baseline', linewidth=linewidth, ls=':', marker='o', markersize=2, color='gray')
    ax.plot(theta, np.append(tmp[:, :, y].mean(axis=0), tmp[:, 0, y].mean(axis=0)) / std[y],
            label='Two-step', linewidth=linewidth, ls='--', marker='x', markersize=4, color='green')
    ax.plot(theta, np.append(ans[:, :, 0, 1, y].mean(axis=0), ans[:, 0, 0, 1, y].mean(axis=0)) / std[y],
            label='Weight', linewidth=linewidth, ls='--', marker='x', markersize=4, color='blue')
    ax.plot(theta, np.append(ans[:, :, 1, 0, y].mean(axis=0), ans[:, 0, 1, 0, y].mean(axis=0)) / std[y],
            label='Discrepancy', linewidth=linewidth, ls='--', marker='x', markersize=4, color='orange')
    ax.plot(theta, np.append(ans[:, :, 1, 1, y].mean(axis=0), ans[:, 0, 1, 1, y].mean(axis=0)) / std[y],
            label='Weight+Discrepancy', linewidth=linewidth, marker='^', markersize=4, color='red')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels))
plt.subplots_adjust(hspace=0.7)
