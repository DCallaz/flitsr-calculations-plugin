from flitsr.merge import Merge  # type:ignore
from flitsr.calculations.perms import Calc
from argparse import ArgumentParser, FileType
from itertools import chain, zip_longest, product
from typing import Dict, List, Tuple, Collection
from numpy import mean, std as stdev
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
import seaborn as sns
from enum import StrEnum, Enum, auto
from collections import defaultdict
from scipy.stats import shapiro, anderson
from dataclasses import dataclass


def rec_dd():
    return defaultdict(rec_dd)


class Type(StrEnum):
    FULL = 'full sampled'
    BASE = ''
    PART = '100 partial sampled'
    STMN = 'steimann'
    BEST = 'best case'
    WRST = 'worst case'
    AVRG = 'average case'


class Result(StrEnum):
    RAW = 'raw'
    ABS_DIFF = 'diff'
    REL_DIFF = 'diff %'


class Measure(StrEnum):
    RESULT = 'result'
    RUNTIME = 'runtime'


class Stat(StrEnum):
    MEAN = 'mean'
    STDEV = 'stddev'

    @property
    def f(self):
        if (self is Stat.MEAN):
            return mean
        elif (self is Stat.STDEV):
            return stdev


@dataclass
class RawData:
    category: str
    calc: Calc
    type: Type
    result: np.ndarray
    runtime: np.ndarray


nums = {Calc.WEFFORT: [1, 2, 3, 4, 5], Calc.RECALL: [1, 5, 10, 20]}
types: Dict[Calc, List[Type]] = {Calc.WEFFORT: list(Type), Calc.RECALL:
                                 [t for t in Type if t is not Type.STMN]}
typ_names = {Type.FULL: 'Enum. (10!)', Type.BASE: 'Framework',
             Type.PART: 'Sampled (100)', Type.STMN: 'Högerle [24]',
             Type.BEST: 'Best case', Type.WRST: 'Worst case',
             Type.AVRG: 'Average case'}
type_order = [Type.FULL, Type.BASE, Type.PART, Type.BEST, Type.WRST, Type.AVRG,
              Type.STMN]
calc_names = {Calc.WEFFORT: ('effort', 'effort'),
              Calc.RECALL: ('recall', 'cutoff')}


def name(type_, calc, fault_num=None, runtime: Measure = Measure.RESULT):
    cname = calc_names[calc][0]
    if (type_ is Type.BASE):
        name = f'{cname}'
    else:
        name = f'{type_} {cname}'
    if (runtime is Measure.RUNTIME):
        name += ' runtime'
    if (fault_num is not None):
        if (calc is Calc.RECALL):
            name += f' at {fault_num}'
        elif (calc is Calc.WEFFORT):
            name += f' ({fault_num})'
        else:
            raise ValueError()
    return name


def time_diff(t1: float, t2: float, perc=True) -> float:
    if (not perc):
        return (t1 - t2)
    if (t1 == 0.0 and t2 == 0.0):
        return 0.0
    elif (t2 == 0.0):
        return 100.0
    elif (t1 > t2):
        return min(100, ((t1 - t2)/t2)*100)
    else:
        return (- (t2 - t1)/t2)*100


def val_diff(v1, v2, perc=True):
    if (not perc):
        return abs(v1 - v2)
    if (v1 == 0 and v2 == 0):
        return 0.0
    elif (v1 == 0 or v2 == 0):
        return 100.0
    elif (v1 < 1e-3 and v2 < 1e-3):
        return 0.0
    else:
        return abs(v1 - v2)*100/v2


def bland_altman_plot(data1, data2, *args, **kwargs):
    # data1 = np.log1p(np.asarray(data1))
    # data2 = np.log1p(np.asarray(data2))
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    abs_diff = (data1 - data2)
    # Difference between data1 and data2
    diff = np.divide(abs_diff, mean, out=np.zeros_like(abs_diff),
                     where=mean != 0)
    md = np.mean(diff)                   # Mean of the difference
    sd = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(data1, diff, s=4, *args, **kwargs)
    # xs = np.geomspace(1e-10, np.max(mean), 100)
    # plt.plot(xs, xs * 0.05)
    # plt.plot(xs, xs * -0.05)
    plt.axhline(md, linestyle='-', *args, **kwargs)
    plt.axhline(md + 1.96*sd, linestyle='--', *args, **kwargs)
    plt.axhline(md - 1.96*sd, linestyle='--', *args, **kwargs)


RawResults = Dict[str, Dict[Calc, Dict[Type, RawData]]]


def get_raw_results(merge: Merge, metrics: Collection[str], modes: Collection[str],
                    calcs: Collection[Calc]) -> RawResults:
    """
    Return a list of the raw results for the given metric, mode and
    calculation.
    """

    raw_results: RawResults = rec_dd()

    # Compress lists of numbers for each stopping point into one (raw) list
    for metric, mode, calc in product(metrics, modes, calcs):
        avgs = merge.avgs[mode][metric]
        for type_ in types[calc]:  # FULL, BASE, PART, (STMN)
            rs: Dict[str, np.ndarray] = {}
            for measure in Measure:  # RESULT, RUNTIME
                chn = chain(*(avgs[name(type_, calc, i, measure)].all
                              for i in nums[calc]))
                rs[measure] = np.asarray(list(chn))
            rd = RawData(metric, calc, type_, **rs)
            raw_results[metric][calc][type_] = rd
    return raw_results


def diff(golden, data):
    mean = np.mean([golden, data], axis=0)
    abs_diff = (data - golden)
    # Difference between data1 and data2
    diff = np.divide(abs_diff, mean, out=abs_diff,
                     where=mean > 1e-1)
    diff *= 100
    return diff


def plot_violins(raw_results: RawResults, categories: Collection[str],
                 calcs: Collection[Calc]):
    colors = ['cyan', 'blue', 'red', 'green', 'magenta', 'yellow']
    cs = len(calcs)
    ms = len(categories)
    plt.rc('font', weight='bold')
    fig, axs = plt.subplots(nrows=cs, ncols=ms, figsize=(9, 3), sharey=True,
                            layout='constrained', squeeze=False)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
    for c, calc in enumerate(calcs):
        for m, category in enumerate(categories):
            cax: plt.Axes = axs[c][m]
            cur = raw_results[category][calc]
            golden = cur[Type.FULL].result
            dataset = []
            all_types = sorted(cur.keys(), key=lambda t: type_order.index(t))
            all_types.remove(Type.FULL)
            all_names = [typ_names[t] for t in all_types]
            for type_ in all_types:
                data = cur[type_].result
                dataset.append(diff(golden, data))
            cax.set_yscale('symlog', linthresh=0.2)
            parts = sns.violinplot(data=dataset, ax=cax, cut=0,
                                   density_norm='area', linewidth=1.7)
            # sns.boxplot(data=dataset, saturation=0.5, width=0.1, whis=[0,100],
            #             boxprops={'zorder': 2, 'facecolor': 'k'}, ax=cax, showfliers=False,
            #             medianprops={'color': 'w'})

            # , facecolor=list(zip(colors, [0.4]*len(colors))), linecolor='black', widths=0.7, showmeans=True, log_scale=True)
            # for pc in parts['bodies']:
            #     pc.set_edgecolor('black')
            #     pc.set_linewidth(.5)
            cax.set_xticks(np.arange(0, len(all_types)), labels=all_names,
                           rotation=45, ha='right')
            cax.set_title(f"{category} {calc_names[calc][1]}".capitalize(),
                          fontsize=14, fontweight='bold')
            cax.grid()
            cax.yaxis.set_tick_params(labelleft=True)
            formatter = FuncFormatter(lambda y, _: '{:g}'.format(y))
            cax.yaxis.set_major_formatter(formatter)
            cax.yaxis.set_ticks([-200, -10, -1, -0.1, 0, 0.1, 1, 10, 200])
            # cax.yaxis.set_minor_locator(AutoMinorLocator())
            cax.yaxis.get_minor_locator().set_params(numticks=np.inf, subs=range(1, 10))
    plt.show()


def calc_stats(merge: Merge, mode: str, metric: str, calc: Calc):
    """
    Calculate all the necessary statistics for the given metric, mode and
    calculation.
    """
    avgs = merge.avgs[mode][metric]

    results: Dict[Type, Dict[Result, Dict[Measure, Dict[Stat, float]]]] = rec_dd()

    # Compress lists of numbers for each stopping point into one (raw) list,
    # and calculate the mean and std-deviation
    raw_calcs: Dict[Type, Dict[Measure, List]] = rec_dd()
    for type_ in types[calc]:  # FULL, BASE, PART, (STMN)
        for measure in Measure:  # RESULT, RUNTIME
            chn = chain(*(avgs[name(type_, calc, i, measure)].all
                          for i in nums[calc]))
            cur_calcs = (np.asarray(list(chn))
                         * (100 if calc is Calc.RECALL else 1))
            raw_calcs[type_][measure] = cur_calcs
            for stat in Stat:  # MEAN, STDEV
                if (measure is Measure.RUNTIME and stat is Stat.STDEV):
                    continue
                results[type_][Result.RAW][measure][stat] = stat.f(cur_calcs)

    # compute the absolute and relative differences to full sampling
    for type_ in types[calc]:  # FULL, BASE, PART, (STMN)
        if (type_ is Type.FULL):
            continue
        for result in [Result.ABS_DIFF, Result.REL_DIFF]:
            # boolean for whether relative diff
            perc = True if (result is Result.REL_DIFF) else False
            for measure in Measure:  # RESULT, RUNTIME
                # get the function for computing the diff
                diff_func = (val_diff if (measure is Measure.RESULT)
                             else time_diff)
                # create the zipped values
                zipped = zip(raw_calcs[type_][measure],
                             raw_calcs[Type.FULL][measure], strict=True)
                # create the diff array
                diffs = [diff_func(v1, v2, perc=perc) for (v1, v2) in zipped]
                for stat in Stat:  # MEAN, STDEV
                    if (measure is Measure.RUNTIME and stat is Stat.STDEV):
                        continue
                    results[type_][result][measure][stat] = stat.f(diffs)
    return results


Results = Dict[str, Dict[str, Dict[Calc, Dict[Type, Dict[Result,
               Dict[Measure, Dict[Stat, float]]]]]]]


def merge_all(recurse):
    """
    Reads in the results files and calculates all statistics
    """
    # read in results
    merge = Merge()
    merge.read_results(recurse)
    metrics = sorted(merge.metrics)
    modes = sorted(merge.modes)

    # results: Results = rec_dd()

    all_calcs: List[Calc] = [Calc.WEFFORT, Calc.RECALL]

    # get all of the raw results
    raw_results = get_raw_results(merge, metrics, modes, all_calcs)

    # code to get the max difference of a particular technique
    # t = raw_results['tarantula']['base'][Calc.RECALL]
    # diffs = diff(t[Type.FULL].result, t[Type.PART].result)
    # zp = zip(diffs, t[Type.PART].result, t[Type.FULL].result,
    #          t[Type.BASE].result)
    # print(sorted(zp, key=lambda x: x[0], reverse=True)[:10])

    # plot the violin plots
    plot_violins(raw_results, metrics, all_calcs)

    # produce table for the run-times
    for calc in all_calcs:
        printed_header = False
        for metric, mode in product(metrics, modes):
            cur = raw_results[metric][calc]
            sort_types = sorted(cur.keys(), key=lambda t: type_order.index(t))
            if (not printed_header):
                print('', *[typ_names[t] for t in sort_types], sep=' & ',
                      end='\\\\\n')
                printed_header = True
            print(metric.capitalize(), *[f'{np.mean(cur[t].runtime):.4f}'
                  for t in sort_types], sep=' & ', end='\\\\\n')

    # produce table for tie info
    class TieInfo(StrEnum):
        FPT = 'faults per critical tie',
        LPT = 'fault locs per critical tie',
        TS = 'critical tie size',
        NT = 'number of critical ties'

    tieinfo_out = {TieInfo.FPT: '$\\obar{f}$', TieInfo.LPT: '$\\obar{l}$',
                   TieInfo.TS: '$\\obar{n}$', TieInfo.NT: '$\\# s$'}

    print('', *tieinfo_out.values(), sep=' & ', end='\\\\\n')
    for metric, mode in product(metrics, modes):
        avgs = merge.avgs[mode][metric]
        print(metric.capitalize(), *[f'{avgs[ti].eval():.2f}' for ti in TieInfo],
              sep=' & ', end='\\\\\n')

    # calculate stats for each metric and mode
    # for metric in metrics:
    #     for mode in modes:
    #         for calc in Calc:  # EFFORT, RECALL
    #             results[mode][metric][calc] = calc_stats(merge, mode,
    #                                                      metric, calc)
    # print_results(results, modes, metrics)


def print_results(results: Results, modes: List[str], metrics: List[str]):
    Rs = list(Result)
    for metric in sorted(metrics):
        for mode in sorted(modes):
            print(f'{mode} {metric}:')
            r = results[mode][metric]
            for calc in sorted(Calc):
                print(f'{calc}')
                print('&', end=' ')
                Ts = [t for t in Type if t in r[calc].keys()]
                print(*Ts, sep=' & ', end='\\\\\n')
                print(*Rs, sep=' & ', end='\\\\\n')
                for measure in sorted(Measure):
                    for stat in sorted(Stat):
                        if (measure is Measure.RUNTIME and stat is Stat.STDEV):
                            continue
                        print(f'{stat} {measure}', end=' & ')
                        for type_ in Ts:
                            for result in Rs:
                                item = r[calc][type_][result][measure][stat]
                                if (isinstance(item, defaultdict)):
                                    continue
                                print(f'{item:.4f}', end=' & ')
                        print('\\\\')


if __name__ == "__main__":
    # get cmd-line args
    parser = ArgumentParser()
    parser.add_argument('-r', '--recurse', action='store_true')
    parser.add_argument('-o', '--output-file', action='store',
                        type=FileType('w'))
    args = parser.parse_args()

    # merge results
    merge_all(args.recurse)
    # import json
    # json.dump(results, args.output_file, indent=2)
