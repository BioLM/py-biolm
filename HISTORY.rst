=======
History
=======

Unreleased
----------

**Pipeline: design primitives for multi-method protein generation**

New config types for ``GenerativePipeline``
(see :mod:`biolmai.pipeline.generative`):

* ``SaturationMutagenesisConfig`` — enumerate all single-mutant variants at
  specified positions, score with any BioLM prediction model, retain top-N.
  Supports structure-aware models via ``pdb_str`` / ``chain``.
* ``IterativeMaskingDMSConfig`` — greedy argmax sequential masking across
  positions using any MLM (ESM2, ESMC).  ``rounds=1`` produces single-point
  variants; ``rounds=2`` produces two-point DMS-style variants.

``GenerativePipeline`` constructor shorthands:

* ``configs=`` — alias for ``generation_configs=``
* ``filters=`` — calls ``add_filter()`` at construction time
* ``data_store=`` — alias for ``datastore=``

``DirectGenerationConfig`` gains a ``label`` field stored in
``generation_metadata`` and surfaced in ``results()`` as ``source_label``.

``BasePipeline.results()`` added as an alias for ``get_final_data()``.

**Breaking change — ``source_label`` column in results():**
``get_final_data()`` / ``results()`` / ``materialize_working_set()`` now
**always** include a ``source_label`` column (``None`` when no label was set).
Code that previously checked ``"source_label" not in df.columns`` must be
updated — the column is unconditionally present.

0.1.0 (2023-09-04)
------------------

* First release on PyPI.
