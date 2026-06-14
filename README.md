#PipeOwl Base & Field Prior

##For COSCUP 2026

This repository demonstrates how to construct a semantic base and a field prior from embedding models.

Outputs
- base.npy — semantic base vectors
- field_prior.npy — reference field prior

## Concept

```word
→ base vector
→ field prior
→ future trained field
```

## What is Base?

A semantic base is a collection of embedding vectors generated from a vocabulary.

word
→ embedding
→ base.npy

The base provides a stable semantic coordinate system for retrieval and similarity search.

## What is Field Prior?

A field prior is an initial semantic direction derived from embedding-space transformations.

For example:

E("happy is defined as ...")
-
E("happy")

The averaged displacement forms a reference direction in semantic space.

The field prior is not a trained field.

Instead, it serves as an initialization or reference coordinate for future field learning, similar in spirit to a positional prior.

##Files

build_base.py
    Build base.npy from a vocabulary list.

build_field.py
    Build field_prior.npy from semantic displacement.

base.npy
    Semantic base vectors.

field_prior.npy
    Reference field prior.
