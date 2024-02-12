<p align="center">
<img src="https://github.com/dobraczka/embarrassment/raw/main/docs/logo.png" alt="kiez logo", width=200/>
</p>
<p align="center">
<a href="https://github.com/dobraczka/embarrassment/actions/workflows/main.yml"><img alt="Actions Status" src="https://github.com/dobraczka/embarrassment/actions/workflows/main.yml/badge.svg?branch=main"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Convenience functions for pandas dataframes containing triples. Fun fact: a group of pandas (e.g. three) is commonly referred to as an [embarrassment](https://www.zmescience.com/feature-post/what-is-a-group-of-pandas-called-its-surprisingly-complicated/).

This library's main focus is to easily make commonly used functions available, when exploring [triples](https://en.wikipedia.org/wiki/Semantic_triple) stored in pandas dataframes. It is not meant to be an efficient graph analysis library.

Usage
=====
You can use a variety of convenience functions, let's create some simple example triples:
```python
>>> import pandas as pd
>>> rel = pd.DataFrame([("e1","rel1","e2"), ("e3", "rel2", "e1")], columns=["head","relation","tail"])
>>> attr = pd.DataFrame([("e1","attr1","lorem ipsum"), ("e2","attr2","dolor")], columns=["head","relation","tail"])
```
Search in attribute triples:
```python
>>> from embarrassment import search
>>> search(attr, "lorem ipsum")
  head relation         tail
0   e1    attr1  lorem ipsum
>>> search(attr, "lorem", method="substring")
  head relation         tail
0   e1    attr1  lorem ipsum
```
Select triples with a specific relation:
```python
>>> from embarrassment import select_rel
>>> select_rel(rel, "rel1")
  head relation tail
0   e1     rel1   e2
```
Perform operations on the immediate neighbor(s) of an entity, e.g. get the attribute triples:
```python
>>> from embarrassment import neighbor_attr_triples
>>> neighbor_attr_triples(rel, attr, "e1")
  head relation   tail
1   e2    attr2  dolor
```
Or just get the triples:
```python
>>> from embarrassment import neighbor_rel_triples
>>> neighbor_rel_triples(rel, "e1")
  head relation tail
1   e3     rel2   e1
0   e1     rel1   e2
```
By default you get in- and out-links, but you can specify a direction:
```python
>>> neighbor_rel_triples(rel, "e1", in_out_both="in")
  head relation tail
1   e3     rel2   e1
>>> neighbor_rel_triples(rel, "e1", in_out_both="out")
  head relation tail
0   e1     rel1   e2
```

Using pandas' [pipe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html) operator you can chain operations.
Let's see a more elaborate example by loading a dataset from [sylloge](https://github.com/dobraczka/sylloge):

```python
    from sylloge import MovieGraphBenchmark

    from embarrassment import clean, neighbor_attr_triples, search, select_rel

    ds = MovieGraphBenchmark()
    # clean attribute triples
    cleaned_attr = clean(ds.attr_triples_left)
    # find uri of James Tolkan
    jt = search(cleaned_attr, query="James Tolkan")["head"].iloc[0]
    # get neighbor triples
    # and select triples with title and show values
    title_rel = "https://www.scads.de/movieBenchmark/ontology/title"
    print(
        neighbor_attr_triples(ds.rel_triples_left, cleaned_attr, jt).pipe(
            select_rel, rel=title_rel
        )["tail"]
    )
    # Output:
    # 12234    A Nero Wolfe Mystery
    # 12282           Door to Death
    # 12440          Die Like a Dog
    # 12461        The Next Witness
    # Name: tail, dtype: object
```


Installation
============
