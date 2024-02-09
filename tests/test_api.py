from itertools import zip_longest

import numpy as np
import pandas as pd
import pytest
from embarrassment import clean, select, select_by_type, select_rel, search
from embarrassment.api import _DEFAULT_RDF_TYPE
from strawman import dummy_df, dummy_triples

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
        '"Rupert Everett"^^@en--ltr',
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
        "Rupert Everett",
        "0",
        "0",
        "2002-01-01",
        "Lorem ipsum dolor sit amet, ^^consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    ]
    tr_df_with_dt["tail"] = values
    return tr_df_with_dt


@pytest.fixture()
def rel_df():
    rdf = dummy_triples(_NUM_TRIPLES, entity_ids=_E_IDS, seed=_SEED)
    type_triples = [
        (eid, _DEFAULT_RDF_TYPE, etype)
        for eid, etype in zip_longest(_E_IDS, ["type1", "type2", "type1"])
    ]
    type_df = pd.DataFrame(type_triples, columns=rdf.columns).fillna("type3")
    return pd.concat([rdf, type_df])


def test_check_triple():
    assert clean(dummy_triples(10)) is not None
    with pytest.raises(ValueError, match="not contain triples"):
        clean(dummy_df((10, 5)))


def test_clean(tr_df_with_dt, tr_df_cleaned):
    assert all(clean(tr_df_with_dt) == tr_df_cleaned)


@pytest.mark.parametrize(
    ("wanted", "hrt", "exp"),
    [
        (
            ("1924-01-01", "2013-01-01", "0.2", "Rupert Everett"),
            "tail",
            ("e1", "e2", "e3", "e4", "e5"),
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
    ("query", "exp_head", "exact"),
    [
        (
            "Rupert Everett",
            ("e4", "e5"),
            True,
        ),
        (
            "Rupert",
            ("e4", "e5"),
            False,
        ),
    ],
)
def test_search(query, exp_head, exact, tr_df_cleaned):
    assert list(search(tr_df_cleaned, query, exact=exact)["head"]) == list(exp_head)
