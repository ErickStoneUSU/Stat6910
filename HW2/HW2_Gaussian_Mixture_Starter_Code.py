# Generating Gaussian Mixture

import random
import numpy as np
from matplotlib.ticker import NullLocator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle
import matplotlib.pyplot as pl
import time
import plotly.graph_objects as go
import pandas as pd


def get_data():
    p_head = 0.7  # Prob. of heads

    def flip(p):
        return 0 if random.random() < (1 - p_head) else 1

    def draw(coin):
        if coin == 1:
            return np.random.normal(1.5, 1, 1)[0]  # Mean 1.5, SD 1
        else:
            return np.random.normal(0, 1, 1)[0]  # Mean 0, SD 1

    flip_res = []  # y
    draw_res = []  # x
    for i in [100, 200, 500, 1000]:
        flip_res.append([[flip(j)] for j in range(i)])
    for i in flip_res:
        draw_res.append([[draw(toss)] for toss in i])  # Draws contains the Gaussian Mixture of Points
    return flip_res, draw_res


def get_ave(li):
    t_1000 = []
    t_500 = []
    t_200 = []
    t_100 = []
    for i in li:
        if '1000' in i:
            t_1000.append(sum(li[i]) / len(li[i]))
        elif '500' in i:
            t_500.append(sum(li[i]) / len(li[i]))
        elif '200' in i:
            t_200.append(sum(li[i]) / len(li[i]))
        elif '100' in i:
            t_100.append(sum(li[i]) / len(li[i]))
    return t_1000, t_500, t_200, t_100


def get_stats(li):
    # error std and error mean
    return np.round(np.std(np.array(1) - li) * 100,2), np.round(np.mean(np.array(1) - li) * 100,2)


def do_it(li):
    li_stat = []
    for i in get_ave(li):
        li_stat.append(get_stats(i))
    return li_stat


def build_models():
    bayes = {}
    logregs = {}
    knns = {}

    start = time.time()
    for i in range(100):
        flips, draws = get_data()
        for y, x in zip(flips, draws):
            # Bayes class
            for ccv in [2, 5, 10, 15, 20, 25]:
                bay = GaussianNB()
                bayes[str(ccv) + '_' + str(len(y))] = cross_val_score(bay, x, y, cv=ccv, scoring="accuracy", verbose=1)

                # log reg
                log_reg = LogisticRegression()
                logregs[str(ccv) + '_' + str(len(y))] = cross_val_score(log_reg, x, y, cv=ccv, scoring="accuracy",
                                                                        verbose=1)

                # knn
                knn = KNeighborsClassifier()
                knns[str(ccv) + '_' + str(len(y))] = cross_val_score(knn, x, y, cv=ccv, scoring="accuracy", verbose=1)

    end = time.time()
    print(end - start)

    pickle.dump(bayes, open('bayes.pck', 'wb'))
    pickle.dump(logregs, open('logregs.pck', 'wb'))
    pickle.dump(knns, open('knns.pck', 'wb'))


def load_models():
    b = pickle.load(open('bayes.pck', 'rb'))
    l = pickle.load(open('logregs.pck', 'rb'))
    k = pickle.load(open('knns.pck', 'rb'))
    return b, l, k


def plot_stats(stats, line_name, color):
    # comes in as a 4 x 2 grid
    z = list(zip(*stats))
    # mean error with std error range
    pl.errorbar([100, 200, 500, 1000], z[1], yerr=z[0], color=color,
                ecolor=color, label=line_name, marker='o', ls='-')


def collect_stats(stats):
    matrix = []
    for i in range(len(stats)):
        for j in zip(*do_it(stats[i])):
            matrix.append(list(j))
    matrix[0].insert(0, 'Bay Std')
    matrix[1].insert(0, 'Bay Err')
    matrix[2].insert(0, 'Log Std')
    matrix[3].insert(0, 'Log Err')
    matrix[4].insert(0, 'KNN Std')
    matrix[5].insert(0, 'KNN Err')
    return matrix


b, l, k = load_models()
# std, mean, error
# axes = pl.gca()
# # axes.set_ylim([0, 1])
# axes.set_xscale('log')
# pl.xlabel('Sample Size Log Scale')
# pl.ylabel('Error %')
# plot_stats(do_it(b), 'Bayes', color='red')
# plot_stats(do_it(l), 'Logistic', color='blue')
# plot_stats(do_it(k), 'KNN', color='green')
# pl.legend(numpoints=1)
# pl.savefig('error.png', transparent=True, bbox_inches='tight', pad_inches=0)
# pl.show()


tabl = collect_stats([b, l, k])
fig, ax = pl.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(tabl, columns=['Type', 100, 200, 500, 1000])

t_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', fontsize=100, bbox=[0,0,1,1])
t_table.auto_set_font_size(False)
t_table.set_fontsize(16)
fig.tight_layout()
pl.gca().set_axis_off()
pl.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                   hspace = 0, wspace = 0)
pl.margins(0, 0)
pl.gca().xaxis.set_major_locator(NullLocator())
pl.gca().yaxis.set_major_locator(NullLocator())
pl.savefig('table_error.png', pad_inches=0)
pl.show()
print('end')

fig = go.Figure(data=go.Scatter(
    x=[1, 2, 3, 4],  # Sample Size in Log Scale
    y=[2, 1, 3, 4],  # Mean Error
    error_y=dict(
        type='data',
        symmetric=False,
        array=[0.1, 0.2, 0.1, 0.1],  # (Max - Mean) Error
        arrayminus=[0.2, 0.4, 1, 0.2])  # (Mean - Min) Error
))

fig.show()

print('end')
# II
# table form
# plot

# COMMENTS: III
