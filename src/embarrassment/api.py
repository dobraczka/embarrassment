import difflib
import functools
from typing import Literal, Sequence, Set, get_args

import pandas as pd

_EXPECTED_LENGTH = 3
_HEAD = "head"
_RELATION = "relation"
_TAIL = "tail"
_DEFAULT_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

InOutBoth = Literal["in", "out", "both"]


def check_triple(func):
    """Raise exception if df is not triple."""

    @functools.wraps(func)
    def wrapper_check(*args, **kwargs):
        num_col = len(args[0].columns)
        if not num_col == _EXPECTED_LENGTH:
            raise ValueError(
                "Cannot perform action if dataframes does not contain triples! Found {num_col} columns..."
            )
        return func(*args, **kwargs)

    return wrapper_check


@check_triple
def clean(attr_df: pd.DataFrame) -> pd.DataFrame:
    def _clean(line) -> str:
        if line is None:
            return ""
        value = str(line).rsplit("^^", 1)[0]
        if value.startswith('"') and value.endswith('"'):
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
    attr_df: pd.DataFrame, query: Sequence[str], hrt: str = _HEAD
) -> pd.DataFrame:
    if len(query) == 1:
        return _select_single(trdf=attr_df, query=query[0], hrt=hrt)
    return attr_df[attr_df[hrt].isin(query)]


@check_triple
def select_rel(trdf: pd.DataFrame, rel: str) -> pd.DataFrame:
    rel_col = trdf.columns[1]
    return _select_single(trdf, query=rel, hrt=rel_col)


@check_triple
def select_by_type(
    rel_df: pd.DataFrame, wanted_type: str, type_rel: str = _DEFAULT_RDF_TYPE
) -> pd.DataFrame:
    rel_col = rel_df.columns[1]
    tail_col = rel_df.columns[2]
    query_exp = f'{rel_col} == "{type_rel}" and {tail_col} == "{wanted_type}"'
    return rel_df.query(query_exp)


@check_triple
def search(attr_df: pd.DataFrame, query: str, exact: bool = False) -> pd.DataFrame:
    val_col = attr_df.columns[2]
    if exact:
        return _select_single(attr_df, query=query, hrt=val_col)
    return attr_df[
        attr_df[val_col].apply(
            lambda x, query: any(difflib.get_close_matches(x, [query])), query=query
        )
    ]


@check_triple
def neighbor_set(
    rel_df: pd.DataFrame, wanted_eid: str, in_out_both: InOutBoth = "both"
) -> Set[str]:
    head_col = rel_df.columns[0]
    tail_col = rel_df.columns[2]
    if in_out_both == "both":
        head_neighbors = set(rel_df[rel_df[tail_col] == wanted_eid][head_col])
        tail_neighbors = set(rel_df[rel_df[head_col] == wanted_eid][tail_col])
        return head_neighbors.union(tail_neighbors)
    if in_out_both == "in":
        return set(rel_df[rel_df[tail_col] == wanted_eid][head_col])
    if in_out_both == "out":
        return set(rel_df[rel_df[head_col] == wanted_eid][tail_col])
    raise ValueError(
        f'Unknown case "{in_out_both}", only {get_args(InOutBoth)} allowed!'
    )
