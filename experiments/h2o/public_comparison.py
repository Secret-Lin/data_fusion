"""Links and caveats for comparing H2O experiments with public methods."""

PUBLIC_COMPARISON = """## Comparison with public methods

- Markley et al., *Averaging Quaternions* (2007): the local weighted quaternion
  mean is a published averaging principle, not a reproduction of an H2O run.
  [NASA paper](https://ntrs.nasa.gov/citations/20070017872).
- H2O (ICCV 2021), H2OTR (CVPR 2023), and QORT-Former (AAAI 2025) report
  single-view RGB object-pose results on the official split. Those numbers are
  not comparable to this project's annotation-consistency control or to a
  FoundationPose RGB-D run until the same input protocol and split are used.
  [H2O paper](https://arxiv.org/html/2104.11181v2),
  [H2OTR](https://github.com/chohoseong/H2OTR),
  [QORT-Former](https://github.com/kcsayem/QORT-Former).
- H2O labels were produced with DenseFusion and multiview ICP; they are not
  five independent tracker measurements. This is why the no-noise control is a
  consistency check rather than an accuracy benchmark.
  [DenseFusion](https://github.com/j96w/DenseFusion).
"""
