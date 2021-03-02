import torch
import matplotlib.pyplot as plt
import numpy as np

# QM9
trial = [i for i in range(10)]
bias = ['000', '100', '010', '001', '110', '101', '011', '111']
alpha = [0.0, 100.0]
beta = [0.0, 0.1]

ans = np.zeros((len(trial), len(bias), len(alpha), len(beta), 12))
for i in range(4):
    for j in range(8):
        for m in range(2):
            for n in range(2):
                path = 'trail({})-bias({})-alpha({})-beta({})'.format(
                    trial[i], bias[j], alpha[m], beta[n]
                )
                ans[i][j][m][n] = torch.load('results/' + path + '_mae.pt').numpy()

tmp = np.zeros((4, 8, 12))
for i in range(4):
    for j in range(8):
        path = 'twostep-trail({})-bias({})'.format(trial[i], bias[j])
        tmp[i][j] = torch.load('results/' + path + '_mae.pt').numpy()

targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
std = np.array([1.4974e+00, 8.1632e+00, 5.9892e-01, 1.2743e+00, 1.2858e+00, 2.8021e+02,
                9.0042e-01, 1.0846e+03, 1.0846e+03, 1.0846e+03, 1.0846e+03, 4.0549e+00])
trial = [0, 1, 2, 3]
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
    ax.plot(theta, np.append(ans[trial, :, 0, 0, y].mean(axis=0), ans[trial, 0, 0, 0, y].mean(axis=0)) / std[y],
            label='Baseline', linewidth=linewidth, ls=':', marker='o', markersize=2, color='gray')
    # ax.plot(theta, np.append(tmp[trial[:3], :, y].mean(axis=0), tmp[trial[:3], 0, y].mean(axis=0)) / std[y],
    #         label='Two-step', linewidth=linewidth, ls='--', marker='x', markersize=4, color='green')
    # ax.plot(theta, np.append(ans[trial, :, 0, 1, y].mean(axis=0), ans[trial, 0, 0, 1, y].mean(axis=0)) / std[y],
    #         label='Weight', linewidth=linewidth, ls='--', marker='x', markersize=4, color='blue')
    # ax.plot(theta, np.append(ans[trial, :, 1, 0, y].mean(axis=0), ans[trial, 0, 1, 0, y].mean(axis=0)) / std[y],
    #         label='Discrepancy', linewidth=linewidth, ls='--', marker='x', markersize=4, color='orange')
    ax.plot(theta, np.append(ans[trial, :, 1, 1, y].mean(axis=0), ans[trial, 0, 1, 1, y].mean(axis=0)) / std[y],
            label='Weight+Discrepancy', linewidth=linewidth, marker='^', markersize=4, color='red')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels))
plt.subplots_adjust(hspace=0.7)


# zinc, lipo, esol, freesolv
def check(name, ts, tall, alp, txt):
    trial = [i for i in range(10)]
    bias = ['000', '100', '010', '001', '110', '101', '011', '111']
    alpha = [0.0, alp]
    beta = [0.0, 0.1]
    ans = np.zeros((len(trial), len(bias), len(alpha), len(beta), 1))
    for i in range(tall):
        for j in range(8):
            for m in range(2):
                for n in range(2):
                    path = name + '-trail({})-bias({})-alpha({})-beta({})'.format(
                        trial[i], bias[j], alpha[m], beta[n]
                    )
                    ans[i][j][m][n] = torch.load('results/' + path + '_' + txt + '.pt').numpy()
    tmp = np.zeros((10, 8, 1))
    for i in range(tall):
        for j in range(8):
            path = name + '-twostep-trail({})-bias({})'.format(trial[i], bias[j])
            tmp[i][j] = torch.load('results/' + path + '_' + txt + '.pt').numpy()

    linewidth = 1
    fig = plt.figure()
    r = np.linspace(0, 2, 9)
    theta = np.pi * r

    ax = plt.subplot(1, 1, 1, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticklabels(bias)
    # ax.set_title('Target ' + str(y + 1), loc='left')
    ax.set_rlabel_position(120)
    ax.plot(theta, np.append(ans[ts, :, 0, 0, 0].mean(axis=0), ans[ts, 0, 0, 0, 0].mean(axis=0)),
            label='Baseline', linewidth=linewidth, ls=':', marker='o', markersize=2, color='gray')
    # ax.plot(theta, np.append(tmp[ts, :, 0].mean(axis=0), tmp[ts, 0, 0].mean(axis=0)),
    #         label='Two-step', linewidth=linewidth, ls='--', marker='x', markersize=4, color='green')
    # ax.plot(theta, np.append(ans[ts, :, 0, 1, 0].mean(axis=0), ans[ts, 0, 0, 1, 0].mean(axis=0)),
    #         label='Weight', linewidth=linewidth, ls='--', marker='x', markersize=4, color='blue')
    # ax.plot(theta, np.append(ans[[0,1,2,3], :, 1, 0, 0].mean(axis=0), ans[[0,1,2,3], 0, 1, 0, 0].mean(axis=0)),
    #         label='Discrepancy', linewidth=linewidth, ls='--', marker='x', markersize=4, color='orange')
    ax.plot(theta, np.append(ans[ts, :, 1, 1, 0].mean(axis=0), ans[ts, 0, 1, 1, 0].mean(axis=0)),
            label='Weight+Discrepancy', linewidth=linewidth, marker='^', markersize=4, color='red')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels))
    # plt.subplots_adjust(hspace=0.7)


# check('zinc', [1], 4, 100.0, 'mae')
# check('Lipo', [0, 2], 4, 10.0, 'rmse')
# check('ESOL', [0, 1, 3], 4, 10.0, 'rmse')
# check('FreeSolv', [0, 1, 2, 3], 4, 10.0, 'rmse')


# sider
alp = 100.0
ts = [2]
trial = [i for i in range(10)]
bias = ['000', '100', '010', '001', '110', '101', '011', '111']
alpha = [0.0, alp]
beta = [0.0, 0.1]
ans = np.zeros((len(trial), len(bias), len(alpha), len(beta), 5))
# for i in range(tall):
for j in range(8):
    for m in range(2):
        for n in range(2):
            path = 'sider' + '-trail({})-bias({})-alpha({})-beta({})'.format(
                trial[2], bias[j], alpha[m], beta[n]
            )
            ans[2][j][m][n] = torch.load('results/' + path + '_' + 'rocauc' + '.pt').numpy()
tmp = np.zeros((10, 8, 5))
# for i in range(tall):
for j in range(8):
    path = 'sider' + '-twostep-trail({})-bias({})'.format(trial[2], bias[j])
    tmp[2][j] = torch.load('results/' + path + '_' + 'rocauc' + '.pt').numpy()

linewidth = 1
fig = plt.figure()
r = np.linspace(0, 2, 9)
theta = np.pi * r

y = 3
ax = plt.subplot(1, 1, 1, projection='polar')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticklabels(bias)
# ax.set_title('Target ' + str(y + 1), loc='left')
ax.set_rlabel_position(120)
ax.plot(theta, np.append(ans[ts, :, 0, 0, y].mean(axis=0), ans[ts, 0, 0, 0, y].mean(axis=0)),
        label='Baseline', linewidth=linewidth, ls=':', marker='o', markersize=2, color='gray')
# ax.plot(theta, np.append(tmp[ts, :, y].mean(axis=0), tmp[ts, 0, y].mean(axis=0)),
#         label='Two-step', linewidth=linewidth, ls='--', marker='x', markersize=4, color='green')
# ax.plot(theta, np.append(ans[ts, :, 0, 1, y].mean(axis=0), ans[ts, 0, 0, 1, y].mean(axis=0)),
#         label='Weight', linewidth=linewidth, ls='--', marker='x', markersize=4, color='blue')
ax.plot(theta, np.append(ans[ts, :, 1, 0, y].mean(axis=0), ans[ts, 0, 1, 0, y].mean(axis=0)) ,
        label='Discrepancy', linewidth=linewidth, ls='--', marker='x', markersize=4, color='orange')
ax.plot(theta, np.append(ans[ts, :, 1, 1, y].mean(axis=0), ans[ts, 0, 1, 1, y].mean(axis=0)),
        label='Weight+Discrepancy', linewidth=linewidth, marker='^', markersize=4, color='red')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels))
# plt.subplots_adjust(hspace=0.7)
