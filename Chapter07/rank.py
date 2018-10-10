from typing import (
    Any, Callable, cast, Iterator, Iterable, List, NamedTuple,
    Sequence, Tuple, TypeVar, Union
)

class Rank_Data(NamedTuple):
    """
    >>> data = {'key1': 1, 'key2': 2}
    >>> r = Rank_Data((2, 7), data)
    >>> r.rank_seq[0]
    2
    >>> r.raw
    {'key1': 1, 'key2': 2}
    """
    rank_seq: Tuple[float]
    raw: Any

K_ = TypeVar("K_")  # 用于计算等级值的可比较的数据类型
Source = Union[Rank_Data, Any]  # 输入数据的结构

def rank_data(
        seq_or_iter: Union[Sequence[Source], Iterator[Source]],
        key: Callable[[Rank_Data], K_] = lambda obj: cast(K_, obj)
    ) -> Iterable[Rank_Data]:
    """分3种情况处理输入数据，第一种情况递归调用自身，
    第2、3种情况将输入数据转化为指定格式赋给 data，然后交给 rerank 处理

    >>> pairs = ((7, 2.3), (21, 1.2), (32, 0.8), (5, 1.2), (11, 18))
    >>> rank_x = tuple(rank_data(pairs, key=lambda x:x[0] ))
    >>> rank_xy = (rank_data(rank_x, key=lambda x:x[1] ))
    >>> list(rank_xy)
    [Rank_Data(rank_seq=(5.0, 1.0), raw=(32, 0.8)), Rank_Data(rank_seq=(1.0, 2.5), raw=(5, 1.2)), Rank_Data(rank_seq=(4.0, 2.5), raw=(21, 1.2)), Rank_Data(rank_seq=(2.0, 4.0), raw=(7, 2.3)), Rank_Data(rank_seq=(3.0, 5.0), raw=(11, 18))]
    """
    if isinstance(seq_or_iter, Iterator):
        # Not a sequence? Materialize a sequence object.
        yield from rank_data(list(seq_or_iter), key)
        return
    data: Sequence[Rank_Data]
    if isinstance(seq_or_iter[0], Rank_Data):
        # Collection of Rank_Data is what we prefer.
        data = seq_or_iter
    else:
        # Collection of non-Rank_Data? Convert to Rank_Data and process.
        empty_ranks: Tuple[float] = cast(Tuple[float], ())
        data = list(
            Rank_Data(empty_ranks, raw_data)
            for raw_data in cast(Sequence[Source], seq_or_iter)
        )

    for r, rd in rerank(data, key):     # 将 rerank 算出的等级值和原有 Rank_Data 组合成新的 Rank_Data
        new_ranks = cast(Tuple[float], rd.rank_seq + cast(Tuple[float], (r,)))
        yield Rank_Data(new_ranks, rd.raw)

def rerank(
        rank_data_iter: Iterable[Rank_Data],
        key: Callable[[Rank_Data], K_]
    ) -> Iterator[Tuple[float, Rank_Data]]:
    """对数据序列排序，调用 ranker 返回等级值

    >>> data = [Rank_Data(rank_seq=(), raw=(7, 2.3)), Rank_Data(rank_seq=(), raw=(21, 1.2)), Rank_Data(rank_seq=(), raw=(32, 0.8)), Rank_Data(rank_seq=(), raw=(5, 1.2)), Rank_Data(rank_seq=(), raw=(11, 18))]
    >>> key = lambda x: x[0]
    >>> list(rerank(data, key))
    [(1.0, Rank_Data(rank_seq=(), raw=(5, 1.2))), (2.0, Rank_Data(rank_seq=(), raw=(7, 2.3))), (3.0, Rank_Data(rank_seq=(), raw=(11, 18))), (4.0, Rank_Data(rank_seq=(), raw=(21, 1.2))), (5.0, Rank_Data(rank_seq=(), raw=(32, 0.8)))]
    """
    sorted_iter = iter(
        sorted(                # 对数据序列排序
            rank_data_iter, key=lambda obj: key(obj.raw)
        )
    )
    # 将数据序列拆成头部和尾部，调用 ranker 返回等级值
    head = next(sorted_iter)
    yield from ranker(sorted_iter, 0, [head], key)

def yield_sequence(
        rank: float,
        same_rank_iter: Iterator[Rank_Data]
    ) -> Iterator[Tuple[float, Rank_Data]]:
    """返回等级值和当前数据项组成的二元组

    >>> rank = 1
    >>> same_rank_seq = [Rank_Data(rank_seq=(), raw=(5, 1.2))]
    >>> list(yield_sequence(rank, iter(same_rank_seq)))
    [(1, Rank_Data(rank_seq=(), raw=(5, 1.2)))]

    >>> rank = 2
    >>> same_rank_seq = [Rank_Data(rank_seq=(), raw=(5, 1.2)), Rank_Data(rank_seq=(), raw=(5, 3.9))]
    >>> list(yield_sequence(rank, iter(same_rank_seq)))
    [(2, Rank_Data(rank_seq=(), raw=(5, 1.2))), (2, Rank_Data(rank_seq=(), raw=(5, 3.9)))]
    """
    head = next(same_rank_iter)
    yield rank, head
    yield from yield_sequence(rank, same_rank_iter)

def ranker(
        sorted_iter: Iterator[Rank_Data],   # 序列尾部
        base: float,
        same_rank_seq: List[Rank_Data],     # 序列头
        key: Callable[[Rank_Data], K_]
    ) -> Iterator[Tuple[float, Rank_Data]]:
    """根据给定的 base 值，在一个排好序的序列上输出等级值计算结果

    >>> key0 = lambda x: x[0]
    >>> rank_data_iter = iter([Rank_Data(rank_seq=(), raw=(7, 2.3)), Rank_Data(rank_seq=(), raw=(21, 1.2)), Rank_Data(rank_seq=(), raw=(32, 0.8)), Rank_Data(rank_seq=(), raw=(5, 1.2)), Rank_Data(rank_seq=(), raw=(11,18))])
    >>> sorted_iter = iter(sorted(rank_data_iter, key=lambda obj: key0(obj.raw)))
    >>> head = next(sorted_iter)
    >>> list(ranker(sorted_iter, 0, [head], key0))
    [(1.0, Rank_Data(rank_seq=(), raw=(5, 1.2))), (2.0, Rank_Data(rank_seq=(), raw=(7, 2.3))), (3.0, Rank_Data(rank_seq=(), raw=(11, 18))), (4.0, Rank_Data(rank_seq=(), raw=(21, 1.2))), (5.0, Rank_Data(rank_seq=(), raw=(32, 0.8)))]
    """
    try:
        value = next(sorted_iter)
    except StopIteration:
        # 整个序列的最后一个批次，调用 yield_sequence 输出结果
        dups = len(same_rank_seq)
        yield from yield_sequence(
            (base+1+base+dups)/2, iter(same_rank_seq))
        return
    if key(value.raw) == key(same_rank_seq[0].raw):
        # 之所以可以只和后面相邻的值比较，是因为输入的序列已经排好序了
        # 若两个相邻值相同，则通过 list 的 '+' 操作符 合并到一个 list 中，递归处理
        # 所以要把头部写成 list 形式才能递归处理
        yield from ranker(
            sorted_iter, base, same_rank_seq + [value], key)
    else:
        # 若与后面相邻的值不同，说明当前批次已结束，做下面两个动作：
        # 第一步：调用 yield_sequence 输出结果
        dups = len(same_rank_seq)
        yield from yield_sequence(
            (base+1+base+dups)/2, iter(same_rank_seq))
        # 第二步：增加 base 值，递归处理后面的数据
        yield from ranker(
            sorted_iter, base+dups, [value], key)

__test__ = {
    'example': '''
>>> scalars= [0.8, 1.2, 1.2, 2.3, 18]
>>> list(rank_data(scalars))
[Rank_Data(rank_seq=(1.0,), raw=0.8), Rank_Data(rank_seq=(2.5,), raw=1.2), Rank_Data(rank_seq=(2.5,), raw=1.2), Rank_Data(rank_seq=(4.0,), raw=2.3), Rank_Data(rank_seq=(5.0,), raw=18)]
'''
}

def test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    test()
    # data = [Rank_Data(rank_seq=(), raw=(7, 2.3)), Rank_Data(rank_seq=(), raw=(21, 1.2)), Rank_Data(rank_seq=(), raw=(32, 0.8)), Rank_Data(rank_seq=(), raw=(5, 1.2)), Rank_Data(rank_seq=(), raw=(11, 18))]
    # key = lambda x: x[0]
    # list(rerank(data, key))
