from itertools import zip_longest

import numpy as np
import pandas as pd
import pytest
from embarrassment import (
    clean,
    neighbor_attr_triples,
    neighbor_rel_triples,
    search,
    select,
    select_by_type,
    select_rel,
)
from strawman import dummy_df, dummy_triples

_DEFAULT_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_NUM_TRIPLES = 10
_SEED = 42
_E_IDS = [f"e{i}" for i in range(_NUM_TRIPLES)]


@pytest.fixture()
def tr_df_with_dt():
    values = [
        np.nan,
        '"1924-01-01"^^<http://www.w3.org/2001/XMLSchema#date>',
        '"2013-01-01"^^<http://www.w3.org/2001/XMLSchema#date>',
        0.2,
        '"Rupert Everett"^^xsd:string',
        "'John Rupert'^^@en--ltr",
        "0^^<http://www.w3.org/2001/XMLSchema#integer>",
        0,
        '"2002-01-01"^^<http://www.w3.org/2001/XMLSchema#date>',
        "Lorem ipsum dolor sit amet, ^^consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.^^<http://www.w3.org/2001/XMLSchema#string>",
    ]
    trdf = dummy_triples(_NUM_TRIPLES, relation_triples=False, seed=_SEED)
    trdf["tail"] = values
    trdf["head"] = _E_IDS
    return trdf


@pytest.fixture()
def tr_df_cleaned(tr_df_with_dt):
    values = [
        "",
        "1924-01-01",
        "2013-01-01",
        "0.2",
        "Rupert Everett",
        "John Rupert",
        "0",
        "0",
        "2002-01-01",
        "Lorem ipsum dolor sit amet, ^^consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    ]
    tr_df_with_dt["tail"] = values
    return tr_df_with_dt


@pytest.fixture()
def rel_df():
    triples = [
        (eid, _DEFAULT_RDF_TYPE, etype)
        for eid, etype in zip_longest(_E_IDS, ["type1", "type2", "type1"])
    ]
    triples.append(("e1", "rel0", "e4"))
    triples.append(("e5", "rel1", "e1"))
    triples.append(("e4", "rel2", "e6"))
    return pd.DataFrame(triples, columns=["head", "relation", "tail"]).fillna("type3")


def test_check_triple():
    tr_df = dummy_triples(10)
    attr_df = dummy_triples(10, entity_ids=list(tr_df["head"]), relation_triples=False)
    assert clean(tr_df) is not None
    assert tr_df.pipe(clean) is not None
    assert clean(attr_df=tr_df) is not None
    assert neighbor_attr_triples(tr_df, attr_df, "e0") is not None
    assert (
        neighbor_attr_triples(rel_df=tr_df, attr_df=attr_df, wanted_eid="e0")
        is not None
    )
    assert neighbor_attr_triples(tr_df, attr_df=attr_df, wanted_eid="e0") is not None

    false_df = dummy_df((10, 5))
    with pytest.raises(ValueError, match="not contain triples"):
        clean(false_df)

    with pytest.raises(ValueError, match="not contain triples"):
        neighbor_attr_triples(false_df, attr_df=attr_df, wanted_eid="e0")
    with pytest.raises(ValueError, match="not contain triples"):
        neighbor_attr_triples(rel_df=false_df, attr_df=attr_df, wanted_eid="e0")
    with pytest.raises(ValueError, match="not contain triples"):
        neighbor_attr_triples(tr_df, attr_df=false_df, wanted_eid="e0")
    with pytest.raises(ValueError, match="not contain triples"):
        neighbor_attr_triples(tr_df, false_df, wanted_eid="e0")


def test_clean(tr_df_with_dt, tr_df_cleaned):
    assert all(clean(tr_df_with_dt) == tr_df_cleaned)


@pytest.mark.parametrize(
    ("wanted", "hrt", "exp"),
    [
        (
            ("1924-01-01", "2013-01-01", "0.2", "Rupert Everett"),
            "tail",
            ("e1", "e2", "e3", "e4"),
        ),
        (
            ("e1", "e3", "e5"),
            "head",
            ("e1", "e3", "e5"),
        ),
        (
            ("bla", "e3"),
            "head",
            ("e3",),
        ),
        (
            ("bla",),
            "head",
            (),
        ),
        (
            "e3",
            "head",
            ("e3",),
        ),
    ],
)
def test_select(
    wanted,
    hrt,
    exp,
    tr_df_cleaned,
):
    assert list(select(tr_df_cleaned, wanted, hrt=hrt)["head"]) == list(exp)


@pytest.mark.parametrize(
    ("wanted", "exp"),
    [
        (
            "type1",
            ("e0", "e2"),
        ),
        (
            "type2",
            ("e1",),
        ),
        (
            "notype",
            (),
        ),
    ],
)
def test_select_by_type(wanted, exp, rel_df):
    assert list(select_by_type(rel_df, wanted)["head"]) == list(exp)


def test_select_rel(rel_df):
    assert len(select_rel(rel_df, "rel0")) > 0


@pytest.mark.parametrize(
    ("query", "exp_head", "method"),
    [
        (
            "Rupert Everett",
            ("e4",),
            "exact",
        ),
        (
            "Rupert",
            ("e4", "e5"),
            "substring",
        ),
        (
            "Rupert",
            ("e4", "e5"),
            "close",
        ),
    ],
)
def test_search(query, exp_head, method, tr_df_cleaned):
    assert list(search(tr_df_cleaned, query, method)["head"]) == list(exp_head)


@pytest.mark.parametrize(
    ("eid", "iob", "filter_self", "exp"),
    [
        (
            "e1",
            "out",
            True,
            {
                ("e4", _DEFAULT_RDF_TYPE, "type3"),
                ("e4", "rel2", "e6"),
            },
        ),
        (
            "e1",
            "out",
            False,
            {
                ("e4", _DEFAULT_RDF_TYPE, "type3"),
                ("e1", _DEFAULT_RDF_TYPE, "type2"),
                ("e4", "rel2", "e6"),
                ("e1", "rel0", "e4"),
            },
        ),
        (
            "e1",
            "in",
            True,
            {
                ("e5", _DEFAULT_RDF_TYPE, "type3"),
            },
        ),
        (
            "e1",
            "in",
            False,
            {
                ("e5", _DEFAULT_RDF_TYPE, "type3"),
                ("e5", "rel1", "e1"),
            },
        ),
        (
            "e1",
            "both",
            True,
            {
                ("e4", _DEFAULT_RDF_TYPE, "type3"),
                ("e5", _DEFAULT_RDF_TYPE, "type3"),
                ("e4", "rel2", "e6"),
            },
        ),
        (
            "e1",
            "both",
            False,
            {
                ("e4", _DEFAULT_RDF_TYPE, "type3"),
                ("e5", _DEFAULT_RDF_TYPE, "type3"),
                ("e1", _DEFAULT_RDF_TYPE, "type2"),
                ("e4", "rel2", "e6"),
                ("e5", "rel1", "e1"),
                ("e1", "rel0", "e4"),
            },
        ),
    ],
)
def test_neighbor_rel_triples(eid, iob, filter_self, exp, rel_df):
    res = neighbor_rel_triples(rel_df, eid, iob, filter_self)
    assert set(res.itertuples(name=None, index=False)) == exp


@pytest.mark.parametrize(
    ("eid", "iob", "exp"),
    [
        (
            "e1",
            "out",
            {"Rupert Everett"},
        ),
        (
            "e1",
            "in",
            {"John Rupert"},
        ),
        ("e1", "both", {"John Rupert", "Rupert Everett"}),
    ],
)
def test_neighbor_attr_triples(eid, iob, exp, rel_df, tr_df_cleaned):
    res = neighbor_attr_triples(rel_df, tr_df_cleaned, eid, iob)
    assert set(res["tail"]) == exp


@pytest.mark.parametrize(
    ("query", "exp_head", "method"),
    [
        (
            "Rupert",
            ("e4", "e5"),
            "substring",
        ),
    ],
)
def test_pipe(query, exp_head, method, tr_df_with_dt, rel_df):
    assert list(
        tr_df_with_dt.pipe(clean).pipe(search, query=query, method=method)["head"]
    ) == list(exp_head)
