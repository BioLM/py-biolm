Boltz API
=========

Boltz-2 provides GPU-accelerated, structure- and affinity-aware protein–ligand cofolding. Given a protein sequence (FASTA) and small-molecule SMILES, it predicts 3D complex poses plus binding likelihood and a continuous affinity value (IC50‑like), with confidence metrics (ipTM/PAE). The model supports method conditioning (X‑ray/NMR/MD), multi-chain template integration, and distance/pocket constraints with optional steering. Inference runs in ≈20 s/ligand on A100/H100 GPUs and approaches FEP accuracy on public benchmarks while being >1000× faster. Use cases include virtual screening, hit discovery, and hit‑to‑lead ranking.


Performance
-----------

- Constraints and resource profile
  - Batch size: 1 request item per GPU execution (schema default)
  - Sequence/token limit: 1,024 tokens total across all chains per request (proteins and nucleic acids tokenized per residue; ligands per atom)
  - GPU targets: NVIDIA A100 80 GB and H100 80 GB; H100 recommended for affinity mode due to 5× diffusion sampling and 200 steps
  - Memory envelope: 20–30 GB for typical protein–ligand jobs (≤1,024 tokens); peaks scale with token count and number of diffusion samples

- Typical runtime and throughput (end-to-end GPU time; excludes network I/O)
  - Structure quick-pass (pose only; sampling_steps=20, diffusion_samples=1): ~3–8 s per complex on H100; ~6–12 s on A100
  - Affinity mode (default; sampling_steps_affinity=200, diffusion_samples_affinity=5): ~18–25 s per complex on H100; ~30–45 s on A100
  - Throughput per GPU-hour (affinity mode): ~140–200 complexes/H100-hour; ~80–120 complexes/A100-hour
  - Scaling: near-linearly with GPU count for embarrassingly parallel batches; production screens routinely run 10–60 concurrent workers

- Accuracy and speed trade-offs (external benchmarks)
  - Hit-to-lead/lead optimization (FEP+ benchmarks)
    - 4-target subset (CDK2, TYK2, JNK1, P38): Pearson R=0.66; ~1,000× faster than ABFE/FEP walltime per analogue
    - OpenFE subset (876 complexes): Pearson R=0.62; pairwise MAE ≈0.93 kcal/mol (assay-averaged)
  - Hit discovery (MF-PCBA)
    - Average Precision=0.0248; EF@0.5%=18.4; AUROC=0.812 (assay-averaged); substantially higher early enrichment than docking (Chemgauss4 EF@0.5%≈2.0)
  - Structure quality
    - Broad PDB 2024–2025: competitive with open co-folding models; antibody–antigen still trails AlphaFold3 but improves vs Boltz‑1
    - MD-local dynamics: improved RMSF correlation vs Boltz‑1; competitive with specialized MD emulators

- Relative performance vs other BioLM models (practical guidance)
  - ESMFold
    - Speed: fastest single-structure monomer modeling (sub‑seconds to a few seconds), but no co-folding with ligands and no affinity
    - When you need end-to-end protein–ligand pose+affinity in one call, Boltz‑2 delivers in tens of seconds with screening-grade accuracy
  - AlphaFold2
    - Accuracy: strong monomer/multimer structures; no native small-molecule co-folding or affinity head
    - Workflow latency: requires separate docking/scoring to approximate affinity; Boltz‑2 provides pose+affinity directly with lower end-to-end time for screening/optimization loops
  - Chai‑1 / ProteinX (co-folders)
    - Speed: similar walltimes for pose sampling; Boltz‑2 adds an integrated affinity module (binary + IC50-like regression) with state-of-the-art screening enrichment
    - Physicality: Boltz‑2x (steering potentials) reduces clashes and improves stereochemistry without sacrificing speed materially
  - NanobodyBuilder / ABodyBuilder
    - Antibody domains: faster and more accurate for nanobody-only structure tasks; Boltz‑2 preferred when small-molecule binding and affinity scoring are required
  - Docking (Chemgauss4 / FRED)
    - Speed: docking is CPU-seconds per compound; accuracy on MF-PCBA materially lower than Boltz‑2 (AP and EF@k); Boltz‑2 better early enrichment at modest extra GPU cost
  - FEP (OpenFE/FEP+/ABFE)
    - Accuracy: FEP remains top on finely curated series; Boltz‑2 approaches FEP’s rank-correlation on public sets while being >1,000× faster, enabling rapid triage and pre-FEP downselection

- What drives runtime and how to tune it
  - Dominant factors: diffusion sampling steps and number of diffusion samples (structure and affinity); token count (≤1,024); ligand atom count
  - Fast presets (typical outcomes; H100)
    - Rapid triage: 20 steps × 1 sample → 3–8 s; best for pocket finding and rough ranking
    - Standard affinity: 200 steps × 5 samples → ~20 s; recommended balance of stability and speed
    - High-precision affinity: 200 steps × 10 samples → ~35–40 s; lowers variance on tight SAR series
  - MSA
    - MMseqs2-based pipelines and caching provide markedly faster alignment retrieval than JackHMMER-style searches; set subsample_msa=True to reduce I/O and minor compute at small accuracy cost

- System optimizations that impact latency/cost
  - Mixed-precision (bfloat16) and optimized triangle attention (“trifast”) in the trunk reduce memory and improve throughput at long token lengths
  - Pocket-centric cropping for affinity limits pairwise compute to protein–ligand and intra‑ligand interactions, yielding >5× reduction in pair feature volume vs whole-complex processing
  - Physics steering potentials (Boltz‑2x) add negligible overhead but improve clash rates and stereochemistry; template/contact/pocket steering available for controllability without retraining
  - Alignment and feature caching amortize overhead across campaigns; cost scales with unique protein sequences rather than compound count in large screens

- Input and output types (relevant to performance and payload size)
  - Input
    - items: list of jobs, each with molecules: List[BoltzEntity]
    - BoltzEntity.type ∈ {protein, dna, rna, ligand}; proteins/nucleic acids require sequence; ligands require SMILES or CCD
    - Optional constraints: bond, pocket, contact (one or more per job) for steering/conditioning
    - Optional templates: one or more CIF templates (mono- or multimeric) mapped to chain_id(s)
    - Params (key latency levers): sampling_steps, diffusion_samples, sampling_steps_affinity, diffusion_samples_affinity, potentials, max_msa_seqs/subsample_msa, seed
  - Output
    - cif: mmCIF text of the predicted complex
    - confidence: scalar and per-chain metrics (ptm, iptm, ligand_iptm, pLDDT/iPlddt, PDE/iPDE)
    - affinity (if enabled and binder specified): 
      - affinity_pred_value: IC50-like regression (log10 scale; optional MW correction)
      - affinity_probability_binary: binding likelihood (0–1)
      - Per‑member ensemble fields: affinity_pred_value{1,2}, affinity_probability_binary{1,2}
    - Optional large arrays (only if requested via include): plddt (per‑token), pae/pde (pairwise), embeddings (single s and pair z). These can be sizable; omit unless needed to minimize transfer time

- Reproducibility and variance
  - seed controls stochastic sampling; increasing diffusion_samples reduces per-assay variance in ranks at near-linear runtime cost
  - Ensemble averaging (two affinity heads) improves robustness and mitigates over-optimization during generative design

- Practical screening guidance (cost vs accuracy)
  - Library triage: run standard affinity (~20 s/H100) across 10^5–10^6 compounds with horizontal scaling; downselect 0.1–1% for FEP or ABFE
  - SAR optimization: high-precision affinity with 10 samples when rank ties matter; still orders of magnitude faster than FEP for iterative cycles


Applications
------------

- High-throughput virtual screening with structure-based affinity scoring: Use Boltz-2’s binding likelihood plus IC50-like affinity value to triage very large purchasable libraries (e.g., Enamine HLL 460k, Kinase 65k) in hours on GPUs, enriching true actives far beyond docking and ipTM heuristics, enabling cheap, early down-selection before wet lab confirmation; this is valuable because it delivers near-FEP ranking quality at ~20 seconds/ligand and showed strong enrichment (e.g., MF-PCBA EF@0.5% ≈ 18.4) and prospective TYK2 success, but results depend on correct pocket/state identification and are less reliable when cofactors, ions, or critical waters are essential and not modeled
- Hit-to-lead and lead optimization ranking within congeneric series: Prioritize analogs by predicted affinity differences to guide SAR cycles on tractable targets (e.g., kinases CDK2, TYK2, JNK1, p38) with performance approaching FEP on public benchmarks (Pearson R ≈ 0.66 on 4-target FEP subset) at >1000× lower cost, helping medicinal chemists focus synthesis on the most promising modifications; this is not optimal when the binding mode is misassigned, the crop misses long-range interactions, or for target classes that historically require custom preparation (e.g., some GPCRs) without additional care
- Generative design scoring loop for synthesizable lead discovery: Integrate Boltz-2 as the reward signal inside a synthesis-aware generator (e.g., SynFlowNet sampling Enamine 76B REAL space) to propose diverse, make-on-demand, high-scoring binders, enabling exploration beyond fixed libraries while keeping chemistry practical; this accelerates design-make-test cycles and yielded strong prospective TYK2 candidates validated by ABFE in silico, but guardrails are needed to prevent reward hacking (use model ensembling, PAINS/med-chem filters, and secondary physics or experimental validation)
- Constraint- and template-steered pose modeling to test binding hypotheses: Apply contact/pocket distance constraints and multimeric template steering to force or bias poses for specific binding modes, enabling fragment-growing directions, warhead placement in covalent campaigns, or site-directed mutagenesis planning in enzymes where pocket geometry drives selectivity; this improves hypothesis testing without retraining, yet remains sensitive to incorrect assumptions—overly aggressive constraints can yield plausible-looking but non-physical poses, so use confidence metrics and, where needed, physics relaxation downstream
- Protein conformational ensemble modeling for pocket-state selection: Use method conditioning (e.g., MD vs X-ray) and ensemble prediction to sample alternative conformations for structure-enabled campaigns, such as selecting open/closed states for SBDD, disambiguating induced-fit risks, or prioritizing constructs for crystallography and cryo-EM; this helps match screening to the relevant biological state and showed improved RMSF correlations versus prior releases, but it is not a substitute for full MD—coverage of rare states and cofactor-dependent rearrangements can be limited without additional experimental or simulation data


Limitations
-----------

- **Batch Size** and **Maximum sequence length**: The ``items`` list supports at most 1 entry (``BoltzModelParams.batch_size = 1``). Total tokenized length across all ``molecules`` in an item must be ≤ ``max_sequence_len = 1024`` (proteins/DNA/RNA tokenize at residue/nucleotide level; ligands at atom level). Exceeding this requires manual cropping or splitting into multiple requests. Runtime scales with sampling: ``recycling_steps`` default ``3`` (range ``1–10``), ``sampling_steps`` default ``20`` (``1–200``), ``diffusion_samples`` default ``1`` (``1–10``); affinity uses ``sampling_steps_affinity = 200`` and ``diffusion_samples_affinity = 5`` by default. Disabling ``potentials = False`` can increase physically implausible poses without reducing the above limits
- Input typing and validations: Each entry in ``molecules`` is a ``BoltzEntity``. For ``type`` = ``"protein"``/``"dna"``/``"rna"``, ``sequence`` is required; for ``"ligand"``, ``sequence`` must be omitted and either ``smiles`` or ``ccd`` is required. ``alignment`` is allowed only for proteins with keys ``"mgnify"``, ``"small_bfd"``, ``"uniref90"``; ``modifications`` are only valid for proteins/DNA/RNA. Optional ``templates`` require non-empty ``cif`` and any ``chain_id`` referenced must match a chain in ``molecules``. Optional ``constraints`` entries must include at least one of ``bond``, ``pocket``, or ``contact``; ``pocket.binder`` supports only a single chain; ``bond`` is supported only between CCD ligands and canonical residues; all chain IDs in constraints must exist in ``molecules``
- Affinity I/O and scope: Affinity outputs are returned only when ``params.affinity`` is provided with a valid ``binder`` chain ID; otherwise the ``affinity`` block in ``results`` is omitted. Returned fields include ``affinity_pred_value`` (IC50-like, assay-dependent scale) and ``affinity_probability_binary``, plus per-model components (``*_1``, ``*_2``). Enabling ``params.affinity_mw_correction = True`` applies a molecular weight calibration to the value head and can change cross-assay comparability. Practical limits: affinity depends on the predicted 3D protein–ligand pose; wrong pocket selection, mis-docked ligands, or incorrect conformational state will degrade predictions. The affinity head does not explicitly model cofactors/ions/water/multimeric partners, and the fixed pocket crop can truncate long-range/allosteric interactions
- Structure/dynamics caveats: While physically-steered predictions improve plausibility, the model can miss large induced-fit motions and remains challenged by very large complexes. MD method conditioning improves local flexibility readouts, but it is not a substitute for molecular dynamics simulations. For antibody–antigen interfaces, performance improves over prior versions yet may lag the strongest specialty or proprietary models on difficult, unseen antigens
- When Boltz-2 is not the best first tool: For ultra-large, early-stage library triage (≫10^6 compounds) under tight time/budget constraints, lightweight sequence filters or docking can be more appropriate up front; use Boltz-2 for structure-aware re-ranking and lead optimization. Projects requiring tightly calibrated absolute ΔG across a narrow chemical series, or difficult targets (e.g., some GPCRs), may still require ABFE/FEP protocols for final ranking. Do not use structure confidence metrics (e.g., ``confidence.iptm``, ``confidence.ptm``) as proxies for affinity—on benchmarks, pose confidence is not predictive of binding strength
- Output size and throughput: Adding optional outputs via ``include`` increases compute and payload size. ``include = ["embeddings"]`` returns high-dimensional ``embeddings.s`` (single) and ``embeddings.z`` (pairwise) tensors; ``include = ["pae","pde","plddt"]`` returns per-token/pair matrices that scale quadratically with token count. Prefer omitting large extras in high-throughput runs. The ``cif`` in ``results`` contains the top-ranked sample; increasing ``diffusion_samples``/``sampling_steps`` raises latency but not response size


How We Use It
-------------

BioLM integrates Boltz-2 as the affinity and structure engine within our design–make–test–learn workflows to accelerate decision-making from hit discovery through lead optimization. We co-fold protein–ligand complexes, apply method, template, and pocket conditioning to test mechanistic hypotheses, and use Boltz-2’s binding likelihood and IC50-like value for assay-aware ranking of large vendor libraries and generative candidates. These scores integrate with our generative chemistries (e.g., GFlowNet-based models), protein design models (e.g., MLM-driven mutagenesis for interface tuning), developability and ADMET filters, and optional ABFE/FEP confirmation before synthesis, enabling scalable prioritization while maintaining lab relevance via standardized APIs and reproducible pipelines.

- Lead optimization and de novo design: Boltz-2-guided generative exploration yields diverse, synthesizable candidates that are ranked, filtered for medicinal chemistry constraints, and cross-checked with physics-based methods where warranted, reducing screening cost and cycle time.  
- Antibody and enzyme programs: Boltz-2 structure predictions and interface metrics guide CDR or active-site variant selection, integrating with sequence embeddings, biophysical property screens (e.g., charge, size, stability), and experimental readouts to drive multi-round maturation.


Related
-------

- ``Chai-1`` – Alternative co-folding model; generate complementary complex poses or multimeric templates to steer Boltz-2 via template conditioning.
- ``ESMFold`` – Fast monomer folding to create apo/domain templates, aiding Boltz-2 pocket localization and multi-chain template integration when no complex structure exists.
- ``ESM-IF1`` – Inverse folding on Boltz-2 poses to propose protein mutations that strengthen pocket interactions or resolve clashes for sequence optimization.
- ``Biotite PDB RMSD`` – Compute RMSD between Boltz-2 poses and references (or across samples) to rank poses and assess conformational stability.


References
----------

- Passaro et al. (2025). `Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction <https://github.com/jwohlwend/boltz>`_. Preprint.

