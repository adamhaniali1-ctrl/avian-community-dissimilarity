# Release checks

The archived research files were preserved. Public entry points were added for reviewers, and calculation functions were separated from the historical runners' stage-specific file inventory checks. The scientific function and class definitions were compared by Python syntax tree against the historical source. The source mapping and machine-readable test results are in `provenance/`.

Verified on 20 November 2026:

- All 459 files in the review archive match recorded SHA256 checksums.
- One complete 36-fit SSI forest selection search (India withheld, primary condition) chose candidate_008, matching the archived selection.
- All 33 event and pair manifests regenerate byte identically.
- All 107,742 evaluation predictions align with their archived pair identities and reconstructed Jaccard targets.
- All 3,648 dataset metric estimates match exactly after preserving full precision when parsing missing values and numeric entries.
- All 1,952 KS comparisons agree within 1.12e-16. The empirical distribution formula is checked independently against SciPy and the saved statistics.
- All 192 descriptive mean signed errors agree with the manuscript revision tables. These are retrospective descriptive calculations, not a newly prespecified outcome.
- All 40 saved Siamese checkpoints reproduce their saved predictions exactly. All eight saved GDM checkpoints reproduce their predictions within 1e-15.
- Baselines and all eight final SSI forests were refitted. Their predictions match the saved results within 6.67e-16. All eight GDM refits match exactly.
- All four primary Siamese ensemble refits match exactly. The India sensitivity ensemble also matches exactly.
- All 440 confidence interval rows were regenerated with the original 2000 event bootstrap draws per group and match exactly. The transfer criterion results also match.
- The numerical figure script completes for all 23 plots. The current editable Figure 1 is included as SVG, PDF and PNG.

Three sensitivity Siamese ensemble refits are not bit identical to their historical reference: maximum absolute prediction differences are 0.00230446 for Malaysia SAFE, 0.0000506401 for Taiwan and 0.000152493 for USA. The unmodified historical training functions produced the same refit state as the reorganized code in the directly checked Malaysia seed 42 case. The original numerical cause remains unverified. No archived predictions, model settings or manuscript results were replaced to conceal this difference. Saved checkpoint inference provides exact recovery of the reported predictions.

One neural fit from each model selection stage was exercised. The preprocessing fit's MAE differed from the historical reference by about 6.00e-10; the architecture B fit's MAE differed by about 8.96e-5. These runs check execution, not complete recovery of the historical selection search. The full neural selection searches have not been rerun. Fresh selection outputs should be compared with the archived decisions; the scripts stop if a completed selection chooses a different setting. The final model settings remain those actually selected for the study.

Historical caveats remain applicable. All four datasets influenced development, including inspection of forest outcomes. The locally locked GDM and Siamese evaluation is not independent external confirmation. LF diagnostics do not establish comparable biological representations across datasets. The temporal diagnostic's historical `same_date` label denotes an exact timestamp match; positive gaps under one day were omitted from its bins. This diagnostic does not determine the main prediction results. The label and original binning are retained to preserve the research record.

The downloadable data archive was also extracted and checked from a fresh copy of the public code.

The release provides execution and numerical verification evidence. It does not certify that the scientific design is free from limitations or that an editor will accept the paper.
