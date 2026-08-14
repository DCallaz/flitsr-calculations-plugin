from functools import partial
from flitsr.tie import Ties, Tie
from flitsr.calculations.bu_model import BUModel
from flitsr.calculations.calc_decorator import calculation, parameter, \
        timing, get_runtime
from flitsr.calculations.exp_values import effort_exp_val, cut_off_exp_val
from flitsr.calculations.perms import exact_method, Calc
from flitsr.calculations.weffort import nth_print_name, check_fault_type
from flitsr.calculations.precision_recall import stop_type, _get_n, \
        print_name as recall_print_name


# <-------------------------- Effort-based calculations ---------------------->

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


@calculation(partial(nth_print_name, name='steimann runtime'),
             "Display the runtime for (steimann) wasted effort to the Nth "
             "fault", "steimann-time")
@parameter('n', type=check_fault_type)
def steimann_rt(ties: Ties, collapse: bool, n: int) -> float:
    return get_runtime('steimann', {'n': n})


@calculation(partial(nth_print_name, name='wasted effort runtime'),
             "Display the runtime for calculating the wasted effort to the "
             "Nth fault", "weffort-time")
@parameter('n', type=check_fault_type)
def weffort_rt(ties: Ties, collapse: bool, n: int) -> float:
    return get_runtime('nth', {'n': n})


@calculation(partial(nth_print_name, name='full sampled effort'),
             "Display the (full sampled) wasted effort to the Nth fault",
             "f-sampled", "full-sampled-wasted-effort")
@parameter('n', type=check_fault_type)
@timing
def full_sampled_effort(ties: Ties, collapse: bool, n: int) -> float:
    return effort_exp_val(ties, min(len(ties.faults), n), weffort=True,
                          collapse=collapse,
                          tie_exp_func=partial(_sampled, bu=ties.bu_model))


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
                          tie_exp_func=partial(_sampled, bu=ties.bu_model,
                                               samples=samples))


def _sampled(tie: Tie, k: int, weffort: bool, collapse=False,
             bu: BUModel = BUModel.PERFECT, samples=None) -> float:
    return exact_method(tie.active_fault_locations(collapse), k,
                        tie.elems(collapse, no_passive=True), Calc.WEFFORT,
                        bu=bu, samples=samples)


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

# <----------------------- Cut-off-based calculations ---------------------->


def _sampled_recall(tie: Tie, p: int, collapse=False, bu: BUModel =
                    BUModel.PERFECT, samples=None) -> float:
    return exact_method(tie.active_fault_locations(collapse), p,
                        tie.elems(collapse, no_passive=True), Calc.RECALL,
                        bu=bu, samples=samples)


@calculation(recall_print_name('full sampled recall'),
             "Display the (full sampled) recall at x",
             "f-sampled-r", "full-sampled-recall")
@parameter('x', type=stop_type)
@timing
def full_sampled_recall(ties: Ties, collapse: bool, x: int) -> float:
    if (len(ties.faults) == 0):
        return 0.0
    n = _get_n(x, ties, collapse=collapse)
    efunc = partial(_sampled_recall, bu=ties.bu_model)
    fault_num = cut_off_exp_val(ties, n, collapse=collapse, tie_exp_func=efunc)
    return fault_num/len(ties.faults)


@calculation(recall_print_name('partial sampled recall'),
             "Display the (partial sampled) recall at x",
             "p-sampled-r", "partial-sampled-recall")
@parameter('x', type=stop_type)
@timing
def partial_sampled_recall(ties: Ties, collapse: bool, samples: int,
                           x: int) -> float:
    if (len(ties.faults) == 0):
        return 0.0
    n = _get_n(x, ties, collapse=collapse)
    efunc = partial(_sampled_recall, bu=ties.bu_model, samples=samples)
    fault_num = cut_off_exp_val(ties, n, collapse=collapse, tie_exp_func=efunc)
    return fault_num/len(ties.faults)


@calculation(recall_print_name('recall runtime'),
             "Display the runtime for calculating the recall at x",
             "recall-time")
@parameter('x', type=stop_type)
def recall_rt(ties: Ties, collapse: bool, x: int) -> float:
    return get_runtime('recall', {'x': x})


@calculation(recall_print_name('full sampled recall runtime'),
             "Display the runtime for (full sampled) recall at x",
             "full-sampled-recall-time")
@parameter('x', type=stop_type)
def full_sampled_recall_rt(ties: Ties, collapse: bool, x: int) -> float:
    return get_runtime('full_sampled_recall', {'x': x})


@calculation(recall_print_name('partial sampled recall runtime'),
             "Display the runtime for (partial sampled) recall at x",
             "partial-sampled-recall-time")
@parameter('x', type=stop_type)
def partial_sampled_recall_rt(ties: Ties, collapse: bool, samples: int,
                              x: int) -> float:
    return get_runtime('partial_sampled_recall', {'samples': samples, 'x': x})
