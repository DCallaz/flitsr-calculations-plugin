from functools import partial
from collections import Counter
from itertools import chain
from flitsr.tie import Ties, Tie
from flitsr.calculations.bu_model import BUModel
from flitsr.calculations.calc_decorator import calculation, parameter, \
        timing, get_runtime
from flitsr.calculations.exp_values import effort_exp_val, cut_off_exp_val
from flitsr.calculations.perms import exact_method, Calc
from flitsr.calculations.weffort import nth_print_name, check_fault_type
from flitsr.calculations.precision_recall import stop_type, _get_n, \
        print_name as recall_print_name


# <========================== Effort-based calculations ======================>


# <--------------------------- Steiman calculations -------------------------->

@calculation(partial(nth_print_name, name='steimann effort'),
             "Display the (steimann) wasted effort to the Nth fault",
             "steimann", "steimann-wasted-effort")
@parameter('n', type=check_fault_type)
@timing
def steimann(ties: Ties, collapse: bool, n: int) -> float:
    return effort_exp_val(ties, min(len(ties.faults), n), weffort=True,
                          collapse=collapse, tie_exp_func=_steimann)


def _steimann(tie: Tie, k: int, weffort: bool, collapse=False) -> float:
    n = tie.len(collapse)
    m = tie.num_faults()
    return (k * (n-m))/(m+1)


@calculation(partial(nth_print_name, name='steimann effort runtime'),
             "Display the runtime for (steimann) wasted effort to the Nth "
             "fault", "steimann-time")
@parameter('n', type=check_fault_type)
def steimann_rt(ties: Ties, collapse: bool, n: int) -> float:
    return get_runtime('steimann', {'n': n})


# <------------------------- Framework calculations ------------------------->

@calculation(partial(nth_print_name, name='effort runtime'),
             "Display the runtime for calculating the wasted effort to the "
             "Nth fault", "weffort-time")
@parameter('n', type=check_fault_type)
def weffort_rt(ties: Ties, collapse: bool, n: int) -> float:
    return get_runtime('nth', {'n': n})


# <-------------------------- Sampled calculations -------------------------->

@calculation(partial(nth_print_name, name='full sampled effort'),
             "Display the (full sampled) wasted effort to the Nth fault",
             "f-sampled", "full-sampled-wasted-effort")
@parameter('n', type=check_fault_type)
@timing
def full_sampled_effort(ties: Ties, collapse: bool, n: int) -> float:
    return effort_exp_val(ties, min(len(ties.faults), n), weffort=True,
                          collapse=collapse, tie_exp_func=_sampled)


def nth_sampled_print_name(name: str, ties: Ties, collapse: bool, n: int,
                           samples: int):
    return f"{samples} {name} ({n})"


@calculation(partial(nth_sampled_print_name, name='partial sampled effort'),
             "Display the (partial sampled) wasted effort to the Nth fault",
             "p-sampled", "partial-sampled-wasted-effort")
@parameter('n', type=check_fault_type)
@timing
def partial_sampled_effort(ties: Ties, collapse: bool, samples: int,
                           n: int) -> float:
    return effort_exp_val(ties, min(len(ties.faults), n), weffort=True,
                          collapse=collapse,
                          tie_exp_func=partial(_sampled, samples=samples))


def _sampled(tie: Tie, k: int, weffort: bool, collapse=False,
             samples=None) -> float:
    return exact_method(tie.active_fault_locations(collapse), k,
                        tie.elems(collapse, no_passive=True), Calc.WEFFORT,
                        bu=tie.fault_identify_nums(collapse), samples=samples)


@calculation(partial(nth_print_name, name='full sampled effort runtime'),
             "Display the runtime for (full sampled) wasted effort to the Nth "
             "fault", "full-sampled-effort-time")
@parameter('n', type=check_fault_type)
def full_sampled_rt(ties: Ties, collapse: bool, n: int) -> float:
    return get_runtime('full_sampled_effort', {'n': n})


@calculation(partial(nth_sampled_print_name, name='partial sampled effort runtime'),
             "Display the runtime for (partial sampled) wasted effort to the "
             "Nth fault", "partial-sampled-effort-time")
@parameter('n', type=check_fault_type)
def partial_sampled_rt(ties: Ties, collapse: bool, samples: int,
                       n: int) -> float:
    return get_runtime('partial_sampled_effort', {'samples': samples, 'n': n})


# <------------------------- Best/worst calculations ------------------------->

def _best_case_w(tie: Tie, k: int, weffort: bool, collapse=False) -> float:
    """ Wasted effort to the k-th fault using best case strategy. """
    # note that the bug understanding model doesn't matter here
    # sanity check: there are at least k faults to localize
    if (tie.num_faults(active=True) >= k):
        # best case -> look at only faulty elements first; no wasted effort
        return 0.0
    else:  # if not, you'll have to look through the whole tie
        return (tie.len(collapse) - tie.num_fault_locs(collapse))


def _worst_case_w(tie: Tie, k: int, weffort: bool, collapse=False) -> float:
    """ Wasted effort to the k-th fault using worst case strategy. """
    # worst case -> always look at all non-faulty elements first
    # note that the bug understanding model doesn't matter here
    return tie.len(collapse) - tie.num_active_fault_locs(collapse)


def _avg_case_w(tie: Tie, k: int, weffort: bool, collapse=False) -> float:
    """ Wasted effort to the k-th fault using average case strategy. """
    # average case -> average of best and worst case (i.e. half non-faulty)
    # note that the bug understanding model doesn't matter here
    return (_best_case_w(tie, k, weffort, collapse) +
            _worst_case_w(tie, k, weffort, collapse))/2


@calculation(partial(nth_print_name, name='best case effort'),
             "Display the wasted effort to the Nth fault using best case tie "
             "resolution", "best-case-effort")
@parameter('n', type=check_fault_type)
@timing
def best_case_weffort(ties: Ties, collapse: bool, n: int) -> float:
    return effort_exp_val(ties, min(len(ties.faults), n), weffort=True,
                          collapse=collapse, tie_exp_func=_best_case_w)


@calculation(partial(nth_print_name, name='worst case effort'),
             "Display the wasted effort to the Nth fault using worst case tie "
             "resolution", "worst-case-effort")
@parameter('n', type=check_fault_type)
@timing
def worst_case_weffort(ties: Ties, collapse: bool, n: int) -> float:
    return effort_exp_val(ties, min(len(ties.faults), n), weffort=True,
                          collapse=collapse, tie_exp_func=_worst_case_w)


@calculation(partial(nth_print_name, name='average case effort'),
             "Display the wasted effort to the Nth fault using average case "
             "tie resolution", "avg-case-effort")
@parameter('n', type=check_fault_type)
@timing
def avg_case_weffort(ties: Ties, collapse: bool, n: int) -> float:
    return effort_exp_val(ties, min(len(ties.faults), n), weffort=True,
                          collapse=collapse, tie_exp_func=_avg_case_w)


@calculation(partial(nth_print_name, name='best case effort runtime'),
             "Display the runtime for (best case) wasted effort to the Nth "
             "fault", "best-case-effort-time")
@parameter('n', type=check_fault_type)
def best_case_effort_rt(ties: Ties, collapse: bool, n: int) -> float:
    return get_runtime('best_case_weffort', {'n': n})


@calculation(partial(nth_print_name, name='worst case effort runtime'),
             "Display the runtime for (worst case) wasted effort to the Nth "
             "fault", "worst-case-effort-time")
@parameter('n', type=check_fault_type)
def worst_case_effort_rt(ties: Ties, collapse: bool, n: int) -> float:
    return get_runtime('worst_case_weffort', {'n': n})


@calculation(partial(nth_print_name, name='average case effort runtime'),
             "Display the runtime for (average case) wasted effort to the Nth "
             "fault", "avg-case-effort-time")
@parameter('n', type=check_fault_type)
def avg_case_effort_rt(ties: Ties, collapse: bool, n: int) -> float:
    return get_runtime('avg_case_weffort', {'n': n})


# <======================= Cut-off-based calculations ======================>


# <------------------------- Framework calculations ------------------------->

@calculation(recall_print_name('recall runtime'),
             "Display the runtime for calculating the recall at x",
             "recall-time")
@parameter('x', type=stop_type)
def recall_rt(ties: Ties, collapse: bool, x: int) -> float:
    return get_runtime('recall', {'x': x})


# <-------------------------- Sampled calculations -------------------------->

def _sampled_recall(tie: Tie, p: int, collapse=False, samples=None) -> float:
    return exact_method(tie.active_fault_locations(collapse), p,
                        tie.elems(collapse), Calc.RECALL,
                        bu=tie.fault_identify_nums(collapse), samples=samples)


def cutoff_sampled_print_name(name: str):
    def fn(ties: Ties, collapse: bool, samples: int, x: int):
        return f"{samples} {name} at {x}"
    return fn


@calculation(recall_print_name('full sampled recall'),
             "Display the (full sampled) recall at x",
             "f-sampled-r", "full-sampled-recall")
@parameter('x', type=stop_type)
@timing
def full_sampled_recall(ties: Ties, collapse: bool, x: int) -> float:
    if (len(ties.faults) == 0):
        return 0.0
    n = _get_n(x, ties, collapse=collapse)
    efunc = _sampled_recall
    fault_num = cut_off_exp_val(ties, n, collapse=collapse, tie_exp_func=efunc)
    return fault_num/len(ties.faults)


@calculation(cutoff_sampled_print_name('partial sampled recall'),
             "Display the (partial sampled) recall at x",
             "p-sampled-r", "partial-sampled-recall")
@parameter('x', type=stop_type)
@timing
def partial_sampled_recall(ties: Ties, collapse: bool, samples: int,
                           x: int) -> float:
    if (len(ties.faults) == 0):
        return 0.0
    n = _get_n(x, ties, collapse=collapse)
    efunc = partial(_sampled_recall, samples=samples)
    fault_num = cut_off_exp_val(ties, n, collapse=collapse, tie_exp_func=efunc)
    return fault_num/len(ties.faults)


@calculation(recall_print_name('full sampled recall runtime'),
             "Display the runtime for (full sampled) recall at x",
             "full-sampled-recall-time")
@parameter('x', type=stop_type)
def full_sampled_recall_rt(ties: Ties, collapse: bool, x: int) -> float:
    return get_runtime('full_sampled_recall', {'x': x})


@calculation(cutoff_sampled_print_name('partial sampled recall runtime'),
             "Display the runtime for (partial sampled) recall at x",
             "partial-sampled-recall-time")
@parameter('x', type=stop_type)
def partial_sampled_recall_rt(ties: Ties, collapse: bool, samples: int,
                              x: int) -> float:
    return get_runtime('partial_sampled_recall', {'samples': samples, 'x': x})


# <------------------------- Best/worst calculations ------------------------->

def _best_worst_faults(tie: Tie, k: int, collapse=False, rev=True) -> float:
    """
    Compute how many faults are found when inspecting only faulty locations (k
    of them). When `rev` is true, inspect elements with the most faults first,
    otherwise inspect elements with the most faults last.
    """
    # inspect the elements with the most faults first/last
    flts_by_loc = [fs for fs in tie.active_fault_locations(collapse).values()]
    flts_by_loc = sorted(flts_by_loc, key=lambda x: len(x), reverse=rev)
    # count how many times each fault is found in the first k elements
    counts = Counter(chain.from_iterable(flts_by_loc[:k]))
    # compare to num locs needed to identify each fault
    expected = tie.fault_identify_nums(collapse)
    num_found = 0
    for fault in counts.keys():
        if (counts[fault] >= expected[fault]):
            num_found += 1
    return num_found


def _best_case_r(tie: Tie, k: int, collapse=False) -> float:
    """ Recall at the k-th element in the tie using best case strategy. """
    return _best_worst_faults(tie, k, collapse, True)


def _worst_case_r(tie: Tie, k: int, collapse=False) -> float:
    num_non_faults = tie.len(collapse) - tie.num_active_fault_locs(collapse)
    # inspect non-faulty elements first
    k = max(k - num_non_faults, 0)
    # short-cut if you've already exhausted your budget
    if (k <= 0):
        return 0.0
    # use the rest of the budget (k) to inspect faulty locations
    return _best_worst_faults(tie, k, collapse, rev=False)


def _avg_case_r(tie: Tie, k: int, collapse=False) -> float:
    return (_best_case_r(tie, k, collapse) + _worst_case_r(tie, k, collapse))/2


@calculation(recall_print_name('best case recall'),
             "Display the (best case) recall at x", "best-case-recall")
@parameter('x', type=stop_type)
@timing
def best_case_recall(ties: Ties, collapse: bool, x: int) -> float:
    if (len(ties.faults) == 0):
        return 0.0
    n = _get_n(x, ties, collapse=collapse)
    fault_num = cut_off_exp_val(ties, n, collapse=collapse,
                                tie_exp_func=_best_case_r)
    return fault_num/len(ties.faults)


@calculation(recall_print_name('worst case recall'),
             "Display the (worst case) recall at x", "worst-case-recall")
@parameter('x', type=stop_type)
@timing
def worst_case_recall(ties: Ties, collapse: bool, x: int) -> float:
    if (len(ties.faults) == 0):
        return 0.0
    n = _get_n(x, ties, collapse=collapse)
    fault_num = cut_off_exp_val(ties, n, collapse=collapse,
                                tie_exp_func=_worst_case_r)
    return fault_num/len(ties.faults)


@calculation(recall_print_name('average case recall'),
             "Display the (average case) recall at x", "avg-case-recall")
@parameter('x', type=stop_type)
@timing
def avg_case_recall(ties: Ties, collapse: bool, x: int) -> float:
    if (len(ties.faults) == 0):
        return 0.0
    n = _get_n(x, ties, collapse=collapse)
    fault_num = cut_off_exp_val(ties, n, collapse=collapse,
                                tie_exp_func=_avg_case_r)
    return fault_num/len(ties.faults)


@calculation(recall_print_name('best case recall runtime'),
             "Display the runtime for (best case) recall at x"
             "fault", "best-case-recall-time")
@parameter('x', type=stop_type)
def best_case_recall_rt(ties: Ties, collapse: bool, x: int) -> float:
    return get_runtime('best_case_recall', {'x': x})


@calculation(recall_print_name('worst case recall runtime'),
             "Display the runtime for (worst case) recall at x"
             "fault", "worst-case-recall-time")
@parameter('x', type=stop_type)
def worst_case_recall_rt(ties: Ties, collapse: bool, x: int) -> float:
    return get_runtime('worst_case_recall', {'x': x})


@calculation(recall_print_name('average case recall runtime'),
             "Display the runtime for (average case) recall at x"
             "fault", "avg-case-recall-time")
@parameter('x', type=stop_type)
def avg_case_recall_rt(ties: Ties, collapse: bool, x: int) -> float:
    return get_runtime('avg_case_recall', {'x': x})
