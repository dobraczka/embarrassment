import difflib
import functools
from typing import Literal, Optional, Sequence, Set, Tuple, Union, get_args, overload

import pandas as pd

_EXPECTED_LENGTH = 3

InOutBoth = Literal["in", "out", "both"]
SearchMethod = Literal["exact", "substring", "close"]


def check_triple(func):
    @functools.wraps(func)
    def wrapper_check(*args, **kwargs):
        def _inner_check(df):
            num_col = len(df.columns)
            if not num_col == _EXPECTED_LENGTH:
                raise ValueError(
                    "Cannot perform action if dataframes does not contain triples! Found {num_col} columns..."
                )

        for value in args:
            if isinstance(value, pd.DataFrame):
                _inner_check(value)
        for value in kwargs.values():
            if isinstance(value, pd.DataFrame):
                _inner_check(value)
        return func(*args, **kwargs)

    return wrapper_check


@check_triple
def clean(attr_df: pd.DataFrame) -> pd.DataFrame:
    """Clean attribute triple values.

    Remove datatype tags and return values as string.

    Args:
      attr_df: Attribute triples.

    Returns:
      Cleaned attribute triples.

    Examples:
        >>> import pandas as pd
        >>> attr = pd.DataFrame([("e1","attr1","'lorem ipsum'^^xsd:string"), ("e2","attr2","dolor")], columns=["head","relation","tail"])
        >>> from embarrassment import clean
        >>> clean(attr)
          head relation         tail
        0   e1    attr1  lorem ipsum
        1   e2    attr2        dolor
    """

    def _clean(line) -> str:
        if line is None:
            return ""
        value = str(line).rsplit("^^", 1)[0]
        if (
            value.startswith('"')
            and value.endswith('"')
            or value.startswith("'")
            and value.endswith("'")
        ):
            return value[1:-1]
        return value

    tail_name = attr_df.columns[2]
    attr_df.loc[:, tail_name] = attr_df[tail_name].fillna("").map(_clean)
    return attr_df


def _select_single(trdf: pd.DataFrame, query: str, hrt: str) -> pd.DataFrame:
    query_exp = f'{hrt} == "{query}"'
    return trdf.query(query_exp)


@check_triple
def select(
    trdf: pd.DataFrame, query: Union[Sequence[str], str], hrt: str = "head"
) -> pd.DataFrame:
    """Select triples containing the queried id(s) in specified column.

    Args:
      trdf: Triple DataFrame.
      query: Query id(s).
      hrt: head, relation or tail column name.

    Returns:
      Triples containing the queried id(s) in specified column.

    Examples:
        >>> import pandas as pd
        >>> rel = pd.DataFrame([("e1","rel1","e2"), ("e3", "rel2", "e1")], columns=["head","relation","tail"])
        >>> from embarrassment import select
        >>> select(rel, "e1")
          head relation tail
        0   e1     rel1   e2
        >>> select(rel, ["e1","e3"])
          head relation tail
        0   e1     rel1   e2
        1   e3     rel2   e1
        >>> select(rel, ["e1","e3"], hrt="tail")
          head relation tail
        1   e3     rel2   e1
    """
    if len(query) == 1:
        query = query[0]
    if isinstance(query, str):
        return _select_single(trdf=trdf, query=query, hrt=hrt)
    return trdf[trdf[hrt].isin(query)]


@check_triple
def select_rel(trdf: pd.DataFrame, rel: str) -> pd.DataFrame:
    """Select triples with specific relation.

    Args:
      trdf: Triple DataFrame.
      rel: Wanted relation.

    Returns:
      Triple DataFrame with specific relation.

    Examples:
        >>> import pandas as pd
        >>> rel = pd.DataFrame([("e1","rel1","e2"), ("e3", "rel2", "e1")], columns=["head","relation","tail"])
        >>> from embarrassment import select_rel
        >>> select_rel(rel, "rel1")
          head relation tail
        0   e1     rel1   e2
    """
    rel_col = trdf.columns[1]
    return _select_single(trdf, query=rel, hrt=rel_col)


@check_triple
def select_by_type(
    rel_df: pd.DataFrame,
    wanted_type: str,
    type_rel: str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
) -> pd.DataFrame:
    """Select triples by type.

    Args:
      rel_df: Triple DataFrame.
      wanted_type: Wanted type.
      type_rel: Type relation.

    Returns:
      Triples with specified type.

    Examples:
        >>> import pandas as pd
        >>> rel = pd.DataFrame([("e1","http://www.w3.org/1999/02/22-rdf-syntax-ns#type","type2"), ("e3", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "type1")], columns=["head","relation","tail"])
        >>> from embarrassment import select_by_type
        >>> select_by_type(rel, "type1")
          head                                         relation   tail
        1   e3  http://www.w3.org/1999/02/22-rdf-syntax-ns#type  type1
    """
    rel_col = rel_df.columns[1]
    tail_col = rel_df.columns[2]
    query_exp = f'{rel_col} == "{type_rel}" and {tail_col} == "{wanted_type}"'
    return rel_df.query(query_exp)


@check_triple
def search(
    attr_df: pd.DataFrame, query: str, method: SearchMethod = "exact"
) -> pd.DataFrame:
    """Search for triples with values in attribute triples.

    Args:
      attr_df: Attribute triples.
      query: Query string.
      method: Search method ("exact", "substring", "close").

    Returns:
      Triples where tail matches query.

    Raises:
        ValueError: if unknown search method.

    Examples:
        >>> import pandas as pd
        >>> attr = pd.DataFrame([("e1","attr1","lorem ipsum"), ("e2","attr2","dolor")], columns=["head","relation","tail"])
        >>> from embarrassment import search
        >>> search(attr, "lorem ipsum")
          head relation         tail
        0   e1    attr1  lorem ipsum
        >>> search(attr, "lorem", method="substring")
          head relation         tail
        0   e1    attr1  lorem ipsum
    """
    val_col = attr_df.columns[2]
    if method == "exact":
        return _select_single(attr_df, query=query, hrt=val_col)
    if method == "close":
        return attr_df[
            attr_df[val_col].apply(
                lambda x, query: any(difflib.get_close_matches(x, [query])), query=query
            )
        ]
    if method == "substring":
        return attr_df[attr_df[val_col].str.contains(query)]
    raise ValueError(
        f"Unknown search method {method}, choose from {get_args(SearchMethod)}"
    )


@overload
def _neighbor_triples(
    rel_df: pd.DataFrame,
    wanted_eid: str,
    in_out_both: Literal["out"],
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    ...


@overload
def _neighbor_triples(
    rel_df: pd.DataFrame,
    wanted_eid: str,
    in_out_both: Literal["in"],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    ...


@overload
def _neighbor_triples(
    rel_df: pd.DataFrame,
    wanted_eid: str,
    in_out_both: Literal["both"] = "both",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ...


def _neighbor_triples(
    rel_df: pd.DataFrame,
    wanted_eid: str,
    in_out_both: InOutBoth = "both",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    head_col = rel_df.columns[0]
    tail_col = rel_df.columns[2]
    head_neighbors = None
    tail_neighbors = None

    if in_out_both not in get_args(InOutBoth):
        raise ValueError(
            f'Unknown case "{in_out_both}", only {get_args(InOutBoth)} allowed!'
        )
    if in_out_both in ("in", "both"):
        head_neighbors = rel_df[rel_df[tail_col] == wanted_eid]
    if in_out_both in ("out", "both"):
        tail_neighbors = rel_df[rel_df[head_col] == wanted_eid]
    return head_neighbors, tail_neighbors


def _neighbor_set_head_tail(
    rel_df: pd.DataFrame, wanted_eid: str, in_out_both: InOutBoth = "both"
) -> Tuple[Set[str], Set[str]]:
    head_n, tail_n = _neighbor_triples(
        rel_df, wanted_eid=wanted_eid, in_out_both=in_out_both
    )

    head_col = rel_df.columns[0]
    tail_col = rel_df.columns[2]
    head_set = set() if head_n is None else set(head_n[head_col])
    tail_set = set() if tail_n is None else set(tail_n[tail_col])
    return head_set, tail_set


@check_triple
def neighbor_set(
    rel_df: pd.DataFrame, wanted_eid: str, in_out_both: InOutBoth = "both"
) -> Set[str]:
    """Get set of neighboring entities.

    Args:
      rel_df: Relation triples.
      wanted_eid: Entity id of which the neighborhood is investigated.
      in_out_both: Whether to look at ("in","out","both") edges.

    Returns:
      Set of neighboring ids.

    Raises:
        ValueError: if unknown in_out_both value.

    Examples:
        >>> import pandas as pd
        >>> rel = pd.DataFrame([("e1","rel1","e2"), ("e3", "rel2", "e1")], columns=["head","relation","tail"])
        >>> from embarrassment import neighbor_set
        >>> neighbor_set(rel, "e1") # doctest: +SKIP
        {'e2', 'e3'}
    """
    head_set, tail_set = _neighbor_set_head_tail(
        rel_df=rel_df, wanted_eid=wanted_eid, in_out_both=in_out_both
    )
    return head_set.union(tail_set)


@check_triple
def neighbor_rel_triples(
    rel_df: pd.DataFrame,
    wanted_eid: str,
    in_out_both: InOutBoth = "both",
    filter_self: bool = True,
) -> pd.DataFrame:
    """Find relation triples of immediate neighbors.

    Args:
      rel_df: Relation triples.
      wanted_eid: Wanted entity id, to search neighborhood.
      in_out_both: Whether to look at ("in","out","both") edges.
      filter_self: Remove triples containing wanted entity id.

    Returns:
      relation triples of immediate neighbors of search entity.

    Raises:
        ValueError: if unknown in_out_both value.

    Examples:
        >>> import pandas as pd
        >>> rel = pd.DataFrame([("e1","rel1","e2"), ("e3", "rel2", "e1"), ("e3", "rel2", "e4"), ("e2", "rel2", "e4")], columns=["head","relation","tail"])
        >>> from embarrassment import neighbor_rel_triples
        >>> neighbor_rel_triples(rel, "e1")
          head relation tail
        2   e3     rel2   e4
        3   e2     rel2   e4
        >>> neighbor_rel_triples(rel, "e1", "in")
          head relation tail
        2   e3     rel2   e4
        >>> neighbor_rel_triples(rel, "e1", "in", filter_self=False)
          head relation tail
        1   e3     rel2   e1
        2   e3     rel2   e4
    """
    neighbor_ids = neighbor_set(
        rel_df=rel_df,
        wanted_eid=wanted_eid,
        in_out_both=in_out_both,
    )
    head_col = rel_df.columns[0]
    tail_col = rel_df.columns[2]
    head_n = rel_df[rel_df[head_col].isin(neighbor_ids)]
    tail_n = rel_df[rel_df[tail_col].isin(neighbor_ids)]
    if filter_self:
        head_n = head_n[head_n[tail_col] != wanted_eid]
        tail_n = tail_n[tail_n[head_col] != wanted_eid]
    return pd.concat([head_n, tail_n])


@check_triple
def neighbor_attr_triples(
    rel_df: pd.DataFrame,
    attr_df: pd.DataFrame,
    wanted_eid: str,
    in_out_both: InOutBoth = "both",
) -> pd.DataFrame:
    """Find attribute triples of neighbor entities of specific entity.

    Args:
      rel_df: Relation triples.
      attr_df: Attribute triples.
      wanted_eid: Wanted entity id of which the neighborhood is used.
      in_out_both: Whether to look at ("in","out","both") edges.

    Returns:
      Attribute triples of neighboring entities.

    Raises:
        ValueError: if unknown in_out_both value.

    Examples:
        >>> import pandas as pd
        >>> rel = pd.DataFrame([("e1","rel1","e2"), ("e3", "rel2", "e1")], columns=["head","relation","tail"])
        >>> attr = pd.DataFrame([("e1","attr1","lorem ipsum"), ("e2","attr2","dolor")], columns=["head","relation","tail"])
        >>> from embarrassment import neighbor_attr_triples
        >>> neighbor_attr_triples(rel, attr, "e1")
          head relation   tail
        1   e2    attr2  dolor
    """
    neighbor_ids = neighbor_set(
        rel_df=rel_df, wanted_eid=wanted_eid, in_out_both=in_out_both
    )
    head_col = attr_df.columns[0]
    return attr_df[attr_df[head_col].isin(neighbor_ids)]
