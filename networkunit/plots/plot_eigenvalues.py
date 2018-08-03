import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.colors as colors
import seaborn as sns


def eigenvalues(EWs, ax=None, bins=50, N=None, B=None,
                     spectra_method='SCREE', ylim=None,
                     color=sns.color_palette()[0]):
    if ax is None:
        fig, ax = plt.subplots()
    left, bottom, width, height = ax.get_position()._get_bounds()
    scaling = .55
    ax.set_position([left, bottom,
                     scaling * width, height])
    axhist = plt.axes([left + scaling * width, bottom,
                       (1-scaling) * width, height])
    axhist.yaxis.tick_right()
    axhist.get_xaxis().tick_bottom()
    axhist.yaxis.set_label_position("right")
    axhist.spines["left"].set_visible(False)
    axhist.spines["top"].set_visible(False)

    eigenvalue_spectra(EWs, ax=ax, method=spectra_method, alpha=.05,
                       color=color)

    ax.invert_xaxis()
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel('')
    ax.yaxis.set_ticks_position('none')
    ax.get_xaxis().tick_bottom()
    ax.yaxis.set_major_formatter(NullFormatter())

    if ylim is None:
        ylim = ax.get_ylim()
    else:
        ax.set_ylim(ylim)

    axhist.set_xlabel('Occurence')
    axhist.set_ylabel('Eigenvalue')

    if type(color) == int:
        color = sns.color_palette()[color]

    EW_hist, edges = np.histogram(EWs, bins=bins, density=False)
    axhist.barh(bottom=edges[:-1], width=EW_hist, height=edges[1]-edges[0],
                color=color, edgecolor='w')
    axhist.set_ylim(ylim)

    if N is not None and B is not None:
        res = 100
        q = B / float(N)
        assert q >= 1
        x_min = (1 - np.sqrt(1. / q)) ** 2
        x_max = (1 + np.sqrt(1. / q)) ** 2

        def wigner_dist(x):
            return q / (2 * np.pi) * np.sqrt((x_max - x) * (x - x_min)) / x

        ev_values = np.linspace(x_min, x_max, res)
        dx = edges[1] - edges[0]
        wigner_values = [wigner_dist(ev) * N * dx for ev in ev_values]
        axhist.plot(wigner_values, ev_values, color='k',
                    label='Marchenko-Pastur Distribution')
        tw_bound = x_max + N ** (-2 / 3.)
        axhist.axhline(tw_bound, color='k', linestyle=':',
                       label='Tracy-Widom Bound')
    else:
        tw_bound = None

        axhist.legend()

    return ax, axhist, tw_bound


def eigenvalue_spectra(EWs, method='SCREE', alpha=.05, ax=None, color='r',
                        mute=False, relative=False):
    """

    :param EWs:
    :param method: 'SCREE', 'proportion', 'broken-stick', 'average-root'
    :param alpha:
    :param ax:
    :param color:
    :return:
    """
    EWs = np.sort(EWs)[::-1]
    if relative:
        total_v = np.sum(abs(EWs))
    else:
        total_v = 1

    if method == 'proportion':
        ### How many EWs can explain (1-alpha)% of the total variance
        pc_count = 0
        cum_var = 0
        while cum_var <= (1-alpha) * total_v:
            cum_var += EWs[pc_count]
            pc_count += 1

    elif method == 'res-variance':
        # ToDo: Can a reasonable residual variance be estimated from sample size?
        pc_count = 0

    elif method == 'broken-stick':
        ### Are EWs larger than the expected values of sorted random values
        N = len(EWs)
        series = [1. / (i+1) for i in range(N)]
        predictor = np.array([total_v / N * np.sum(series[k:])
                              for k in range(N)])
        pc_count = np.where(EWs < predictor)[0][0]

    elif method == "average-root":
        ### Are EWs larger than Tr(C)/N=1
        pc_count = len(np.where(EWs > 1)[0])

    elif method == "SCREE":
        a = - EWs[0] / len(EWs)
        b = EWs[0]
        a_s = - 1. / a
        def cut(pc_count):
            b_s = EWs[pc_count] - a_s*pc_count
            x_s = (b_s-b) / (a-a_s)
            y_s = (a*b_s - a_s*b) / (a-a_s)
            # distance from EW to line
            return np.sqrt((pc_count-x_s)**2 + (EWs[pc_count]-y_s)**2)

        pc_count = 0
        prev_distance = 0
        current_distance = cut(pc_count)
        while current_distance >= prev_distance \
          and pc_count < len(EWs)-1:
            pc_count += 1
            prev_distance = current_distance
            current_distance = cut(pc_count)

    if ax:
        def alpha(color_inst, a):
            if type(color_inst) == str:
                if color_inst[0] == '#':
                    color_inst = colors.hex2color(color_inst)
            return [el + (1. - el) * a for el in color_inst]

        mask = np.zeros(len(EWs), np.bool)
        mask[:pc_count] = 1
        ax.plot(np.arange(len(EWs)), abs(EWs)/total_v, color=color)
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=mask, color=alpha(color,.5))
        if pc_count - 1:
            mask[pc_count-1] = 0
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=np.logical_not(mask), color=alpha(color,.9))
        ax.set_xlabel('Eigenvalue #')
        ax.set_ylabel('rel. eigenvalue')
        ax.set_xlim(0, len(EWs))
        ax.set_ylim(0, np.ceil((max(EWs)/total_v)*10)/10.)

    return pc_count
