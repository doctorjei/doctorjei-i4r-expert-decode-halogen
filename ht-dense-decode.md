# Halogen ht Dense Decode Specification
> Version 1.0 (draft 3)

## Overview
This is a functional description of how halogen, Peonist LLC's inference engine, runs the dense matrix-vector products of a model during decode, for weights stored in the **ht** format: 4-bit trellis codes in 16 x 16 tiles, in a rotated (128-point Hadamard) basis. It is a companion to the [i4r expert decode specification](README.md), which covers the routed experts of the same model.

**Methods.** Same as the i4r specification: clean-room style; one actor studied the engine and wrote this description, and a second actor reviewed it to ensure no 'bleed over' into the clean specification. The description is limited to prose, tables of numbers and data-flow descriptions: no code, pseudo-code mirroring the engine, disassembly, instruction listings, identifiers, strings or hex. (The halogen license does not prohibit reverse engineering but does ask that derived source or kernels not be redistributed as one's own work; this document contains neither.)

**Numbers.** As in the i4r specification, absolute timings and throughput are omitted, per request in the halogen license. Times are given as shares of the per-token GPU busy time or relative to another launch, and bandwidth as a share of the GPU's published peak memory bandwidth (256-bit LPDDR5X-8000). Structural facts (launch counts, work splits, register and memory use, data sizes) are given exactly.

**Scope.** The engine running the model's second-version (v2) weights on a gfx1151 GPU, decode only. Covered matrices: the attention and linear-attention projections, the output head, and the three shared-expert projections, all stored as ht with 4-bit codes. The storage format itself is public (halogen's published format specification) and is not restated here.

## 1. Summary

During single-token decode the engine reads its 4-bit ht weights at 72 to 78 percent of the GPU's
published peak bandwidth on the grouped projections and the output head, 59 percent on the output
projection, and 62 percent over all ht work. Five techniques explain this:

1. Few, large launches. All ht projections that read the same input share one launch. A
   linear-attention layer computes its query/key/value and output-gate projections in one launch.
   A full-attention layer computes query, key and value in one launch. The output projection is a
   second launch, and the output head is one more launch per token.
2. Fused rotations and a self-finishing reduction. Each workgroup rotates its own slice of the
   input into the weights' basis on the fly. The workgroup that finishes a 128-row band last sums
   the partial results and applies the output-side rotation and magnitudes. No separate rotation,
   reduction or format-conversion launch exists.
3. Whole-tile, fully coalesced loads with one step of look-ahead. Every load instruction of a
   wavefront covers one complete 128-byte tile. The loads for the next step are in flight while the
   current step's arithmetic runs.
4. The codebook computed in registers. Each value is decoded arithmetically, with the final
   scaling done two values at a time in packed half precision. There is no table and no
   shared-memory lookup. Decoded values feed paired half-precision dot products with 32-bit
   accumulation.
5. Reuse of everything read. A lane reads its eight input values from shared memory once and
   uses them for eight output rows. In verify batches of up to 8 tokens, each weight is decoded
   once and applied to every token.

## 2. Launch structure (per decode token)

| Stage | Launches per layer | Launches per token | What one launch covers |
|---|---|---|---|
| Input projections, linear-attention layer | 1 | 36 | Query/key/value projection (10,240 rows) and output-gate projection (6,144 rows) together, with input rotation and output rotation fused |
| Input projections, full-attention layer | 1 | 12 | Query-and-gate (12,288 rows), key (512 rows) and value (512 rows) together, rotations fused |
| Output projection, all layers | 1 | 48 | One 2,560 x 6,144 matrix (the linear-attention or attention output projection), rotations and the cross-workgroup sum fused |
| Indexer projection, full-attention layers | 1 | 12 | One 640 x 2,560 matrix, same routine as the output projection |
| Router launch, which also holds the shared expert's gate and up projections | 1 | 48 | Covered by the [i4r spec](README.md); the ht part is the shared expert's gate and up (2 x 640 rows) |
| Shared expert's down projection | 0 | 0 | Done inside the routed-expert gate-and-up launch at batch 1 (i4r spec, section 10) |
| Output head | (none) | 1 | 248,320 x 2,560, rotations fused, 32-bit float output |
| ht total | 3 per layer, plus 1 more in full-attention layers | 157 | About 24 percent of the token's roughly 650 launches |

- Projections sharing a launch: every ht matrix that reads the same input vector in the same
  step. A grouped launch holds up to four matrices. Each matrix gets its own contiguous range of
  workgroups and its own split of the input dimension (section 4).
- Fused in, inside the matrix launches: the input-side rotation-vector multiply, the input-side
  128-point Hadamard transform, the conversion of the input to half precision, the cross-workgroup
  sum, the output-side Hadamard transform, the output-side rotation-vector multiply, and the
  rounding of the result to bf16 (projections) or its store as 32-bit floats (output head).
- Grouped launches can also write the rows of one matrix to two destination buffers, split at
  row 256 of each 512-row group. This is consistent with separating the query and gate halves of
  each attention head as they are stored (medium confidence), so no reshuffle launch is needed.
- Not fused into the ht launches: normalisation and residual adds. The input arrives already
  normalised as a bf16 vector, and the output is stored without reading a residual (high
  confidence for the matrix launches read; the neighbouring launches were not part of this scope).
- An optional fused input stage multiplies the input element by element by the SiLU of a second
  vector before rotation. In this decode it is used only for the shared expert's down projection
  at verify batches of 6 and 8 tokens (section 8).
- Shared-expert gate (the scalar, not ht): see section 10 of the [i4r spec](README.md). It is computed at the
  end of the router launch and applied when the shared expert's output is added.
- Output head: it runs the same single-matrix routine as the output and indexer projections, in a
  separate build. Functionally it differs in three ways: the input dimension is not split (one
  workgroup per 128-row band, 20 steps per wavefront, so no partial-sum step), the result is stored
  as 32-bit floats instead of being rounded to bf16, and the launch is far larger (1,940
  workgroups). Nothing is fused after the matrix product: token selection runs afterwards as its
  own two launches (a partial maximum search over slices of the output, then a final pass that
  picks the winner). No sampling step appears in the measured decode.
- Graph capture: no. The API trace of the latest run has no graph calls; every launch is an
  ordinary launch from the host (same as the i4r spec).

## 3. Trellis decoding in parallel

- Unit of work: one lane decodes one 32-bit word of a tile, which holds 8 values at 4 bits each.
  Per step, one lane decodes one word in each of 8 tiles: 64 values per lane per step, and 2,048
  values (8 tiles, 1 KiB of codes) per wavefront per step.
- Getting the 16-bit states: each lane loads its own word and the word just before it in the same
  tile. The first word of a tile takes the tile's last word as its predecessor, which closes the
  circular stream. Joined, the two words contain the eight overlapping 16-bit states of the lane's
  eight values. No cross-lane exchange is needed for the states.
- The predecessor word is fetched with an ordinary load, not passed between lanes. Those loads
  touch the same 128-byte tile as the wavefront's own-word loads, so they are served from cache.
  DRAM traffic stays at one read of each tile (inferred from the access pattern, not measured).
- Codebook: evaluated arithmetically per value in registers, with no table, no shared memory and
  no per-tile constants. Its final scaling is done two values at a time in packed half precision.
- Output of decoding: exact half-precision values, packed two per register, ready for the paired
  dot product.

## 4. Work decomposition

- Wave size 32. Workgroup size 256 (8 wavefronts) in every ht routine.
- One workgroup owns one 128-row band (8 row tiles) and a contiguous slice of the input dimension,
  counted in 128-column blocks. Its 8 wavefronts split that slice: each takes 16-column tile
  columns in turn and covers all 128 rows of the band.
- Within a wavefront: two lanes share each row in a step, each covering half of the tile's
  columns. Each lane accumulates 8 output rows (one row in each of the 8 row tiles), so one
  wavefront produces partial sums for all 128 rows.
- Reduction, in three levels: (a) the two lanes that share a row add their sums with a cross-lane
  exchange, (b) the 8 wavefronts add theirs through shared memory, giving 128 sums per workgroup,
  (c) when the input dimension is split, partial sums go through memory and the last workgroup to
  finish a band reduces them and completes the band (output rotation, magnitudes, store) in the
  same launch, with no floating-point atomics and no second launch.

| Matrix (rows x columns) | Launch | Workgroups for this matrix | Input split (workgroups per band) | 128-column blocks per workgroup |
|---|---|---|---|---|
| Linear-attention query/key/value, 10,240 x 2,560 | grouped, 752 workgroups in total | 560 | 7 | 3 (the last split gets 2) |
| Linear-attention output gate, 6,144 x 2,560 | same launch | 192 | 4 | 5 |
| Attention query and gate, 12,288 x 2,560 | grouped, 832 workgroups in total | 672 | 7 | 3 (the last split gets 2) |
| Attention key, 512 x 2,560 | same launch | 80 | 20 | 1 |
| Attention value, 512 x 2,560 | same launch | 80 | 20 | 1 |
| Output projection, 2,560 x 6,144 | single | 160 | 8 | 6 |
| Indexer projection, 640 x 2,560 | single | 100 | 20 | 1 |
| Output head, 248,320 x 2,560 | single | 1,940 | 1 (no split, no partial-sum step) | 20 |
| Shared-expert down, 2,560 x 640 (verify batches of 6 or 8 only) | single | 100 | 5 | 1 |

- How the split is chosen: small matrices are split finely (20 ways), large ones 4 to 8 ways, so
  that every launch has several hundred workgroups (100 to 1,940). The splits are passed in by the
  host per matrix; they are not fixed in the GPU code.
- Steps per wavefront are short: 1 to 3 tile columns for most projections, 6 for the output
  projection, 20 for the output head. That is why the look-ahead loads (section 5) and the
  self-finishing reduction matter more than a long steady-state loop would.

## 5. Memory access

- Codes: each load is 4 bytes per lane. The 32 lanes of a wavefront read the 32 consecutive words
  of one 128-byte tile, so every load instruction is one fully coalesced 128-byte access. A
  wavefront issues 8 such loads per step for its own words (the 8 row tiles of one tile column,
  which are consecutive in memory: 1 KiB contiguous) plus 8 for the predecessor words.
- Bytes per lane per step: 32 bytes of codes it decodes, 64 bytes requested in total.
- Prefetch: one step of software pipelining. The 16 loads per lane for the next step are issued
  before the arithmetic of the current step starts.
- Codes do not go through shared memory.
- Input vector: read from memory in bf16, 4 values per lane per 128-column block. Each workgroup
  reads only its own slice. The slice is rotated (section 7), rounded to half precision, and kept
  in shared memory for the whole workgroup: 2 bytes per value, up to 2,560 values (5 KiB) per
  token.
- Input reuse: in each step, a lane reads its 8 input values (16 bytes) from shared memory in one
  access and uses them for all 8 of its rows.
- Output: 2 bytes per value (bf16) for projections, 4 bytes (f32) for the output head. Each output
  value is written once, by the finishing workgroup of its band.

## 6. Arithmetic

| Quantity | Precision |
|---|---|
| Decoded weight values | Half precision, exact (two per register) |
| Input after rotation | Half precision (rounded from 32-bit) |
| Products and per-lane accumulation | Paired half-precision dot product added into a 32-bit float accumulator |
| Lane, wavefront and workgroup sums, scratch partials | 32-bit float |
| Output rotation and magnitudes | 32-bit float |
| Stored result, projections | bf16, round to nearest even |
| Stored result, output head | 32-bit float |

- Rotation-vector magnitudes: applied once per vector element, never per weight value or per
  tile. The input-side vector multiplies each input value once during staging, before the
  Hadamard transform. The output-side vector multiplies each output value once after the output
  Hadamard transform, together with the 1/sqrt(128) factor.
- Techniques: the codebook's byte addition is done by an instruction that sums the four bytes of
  a word; decoding and the dot product use packed half-precision arithmetic (section 3). Cross-lane
  exchanges within the wavefront carry the Hadamard transforms and the two-lane row sum.

## 7. The 128-point rotation

| Where | What is applied | How often |
|---|---|---|
| Input side, every ht matrix launch | Input-side rotation vector, then a 128-point Hadamard transform in registers within one wavefront (4 values per lane, 32 lanes), then 1/sqrt(128), then rounding to half precision into shared memory | Once per workgroup for each 128-column block of its slice |
| Output side, every ht matrix launch | 128-point Hadamard transform in registers (8 values per lane, 16 lanes), 1/sqrt(128), output-side rotation vector, rounding to bf16 | Once per 128-row band, in the workgroup that finishes the band |

- No separate rotation launch runs in decode for these matrices. The engine contains standalone
  rotation routines, and the grouped routine has an optional extra range of workgroups for
  rotating a vector, but neither is used in the decode steps measured.
- Sharing between projections that read the same input: the input rotation is not shared through
  memory. Every workgroup re-does it for its own slice. In the linear-attention grouped launch,
  for example, the 2,560-value input is rotated in 2,560 block-transforms across 752 workgroups,
  where 20 would suffice. This repeated work is small arithmetic on data already in registers.
  It removes a launch and a round trip through memory. Each matrix applies its own input-side
  vector (medium confidence: the vector is reached through each matrix's own parameters).
- The output rotation is per band of each matrix and cannot be shared.

## 8. Measurements

Method: kernel trace of the latest run. Decode with the drafter off was separated from the rest by
splitting the trace at idle gaps over 3 ms and labelling each piece by its routine mix and step
count. Two clean drafter-off pieces were used: 63 steps (short prompt) and 36 steps (list prompt).
The two agree within 1 percent; the 63-step figures are shown. Wall-clock time per token is about
109 percent of GPU busy time.

Bandwidth is derived, not counted by hardware: weight bytes from the public format and the model's
matrix shapes (4 bits per value; rotation vectors, under 0.1 percent, excluded) divided by the
measured kernel time. Percentages are of the published peak bandwidth.

| Kernel role (described, not named) | Launches per token | Time per launch, relative to the output projection | Share of GPU busy time per token | Weight bytes per launch | Share of peak bandwidth | Registers per lane / LDS per workgroup |
|---|---|---|---|---|---|---|
| Grouped input projections, linear-attention layer | 36 | 2.15 | 16.3 % | 20.97 MB | 74 % | 66 / 5 KiB |
| Grouped input projections, full-attention layer | 12 | 1.79 | 4.5 % | 17.04 MB | 72 % | 66 / 5 KiB |
| Output projection | 48 | 1.00 | 10.1 % | 7.86 MB | 59 % | 66 / 5 KiB |
| Indexer projection | 12 | 0.20 | 0.5 % | 0.82 MB | 31 % | 66 / 5 KiB |
| Router launch with shared-expert gate and up | 48 | 0.67 | 6.7 % | 1.64 MB of ht (plus the router's own weights) | 18 % (ht bytes only; lower bound) | 95 / 14 KiB |
| Output head | 1 | 30.8 | 6.5 % | 317.85 MB | 78 % | 66 / 5 KiB |
| All ht work | 157 | | 44.6 % | 1,743 MB per token | 62 % | |

- The shared expert's down projection (0.82 MB per layer) runs inside the routed-expert launch at
  batch 1 and is counted in the i4r spec, not here.
- Large launches reach 72 to 78 percent of peak; the small matrices (indexer, shared gate and up)
  are limited by launch and reduction overhead rather than by bandwidth.
- Occupancy: 8 wavefronts per workgroup, no register spills, 5 KiB of shared memory per workgroup
  per token. Achieved occupancy was not measured.
- Theoretical occupancy (derived, not measured). Assumptions from public AMD and LLVM
  documentation: a register file of 1,536 registers per lane per SIMD in wave32, allocated in
  steps of 24; at most 16 wavefronts per SIMD; 128 KiB of shared memory per workgroup processor
  (4 SIMDs).

| Configuration | Registers (allocated) | Shared memory per workgroup | Wavefronts per SIMD, register limit | Wavefronts per SIMD, shared-memory limit | Limit |
|---|---|---|---|---|---|
| ht routines, 1 token | 66 (72) | 5 KiB | 21, capped at 16 | 50 | 16, hardware maximum |
| Router launch with shared gate and up | 95 (96) | 14 KiB | 16 | 18 | 16, hardware maximum |
| ht routines, 8 tokens | 150 (168) | 40 KiB | 9 | 6 | 6, shared memory |

  If the register file were 1,024 per lane instead, allocated in steps of 16 (66 rounds up to 80),
  the 1-token limit would be 12 wavefronts per SIMD. In either case single-token decode is not occupancy-limited by registers or shared memory.
- Test conditions: the model's v2 weights on gfx1151, two short prompts, 64 generated tokens each, drafter off for the figures above.

### Small batches (MTP verify)

| Kernel role | 1 token | 2 tokens | 3 or 4 tokens | 5 tokens | 6 tokens | 8 tokens |
|---|---|---|---|---|---|---|
| Grouped input projections, linear-attention layer | 1.00 | 0.95 | 0.97 | 1.09 | 1.15 | 1.33 |
| Grouped input projections, full-attention layer | 1.00 | 0.94 | 0.99 | 1.14 | 1.20 | 1.40 |
| Output projection | 1.00 | 1.02 | 1.13 | 1.08 | 1.13 | 1.18 |
| Indexer projection | 1.00 | 1.08 | 1.27 | 1.69 | 1.70 | 1.90 |
| Router launch with shared-expert gate and up | 1.00 | 0.95 | 1.11 (3) / 1.16 (4) | 1.29 | 1.37 | 1.75 |
| Output head | 1.00 | 1.00 | 1.00 | 1.02 | 1.04 (one launch) | 1.14 |
| Shared-expert down as its own launch | (in the routed-expert launch) | same | same | same | 0.25 * | 0.28 * |

Median time per launch over the whole run, by batch size, relative to the same role at 1 token
(\* relative to the 1-token output projection, since there is no 1-token launch). The batch-1 column
mixes drafter-off and drafter-on steps; its medians match the drafter-off figures above. Columns 2
to 8 come only from drafter-on (verify) steps and are marked as mixed-segment figures in that sense.

- Weights are read once for all tokens: one launch per projection regardless of batch size, with
  the same grid, the same split per matrix and the same number of workgroups. Each decoded weight
  pair is dotted with every token's input. The output head costs 14 percent more for 8 tokens than
  for 1.
- What changes with batch size: each token gets its own staged input slice in shared memory (5 KiB
  per token) and its own set of 8 accumulators per lane. Registers per lane grow from 66 (1 token)
  to 78, 101, 114, 123 and 150 (2, 4, 5, 6 and 8 tokens). Separate builds exist for 1, 2, 4, 5, 6
  and 8 tokens; a batch of 3 runs in the 4-token build.
- At 6 and 8 tokens the shared expert's down projection leaves the routed-expert launch and runs
  as a separate ht launch: 20 row-bands x 5 input splits, with the SiLU-gated input stage fused in.

## 9. Open questions and confidence

| Item | Basis | Confidence |
|---|---|---|
| Launch grouping, split per matrix, workgroup counts | Static reading plus recorded launch parameters | High |
| Tile-per-load coalescing, 64 values per lane per step, predecessor-word technique | Static reading | High |
| Predecessor loads served from cache (DRAM reads each tile once) | Inferred from the access pattern; no traffic counter | Medium |
| Codebook evaluated arithmetically per value, no table | Static reading | High |
| Precisions (section 6) | Static reading | High |
| Rotations fused at both ends, input rotation repeated per workgroup | Static reading; no rotation launches in the decode trace | High |
| Each matrix uses its own input-side vector | Parameters reached per matrix; values not compared | Medium |
| Two-buffer output split separates query and gate halves | Static reading of the store step; purpose inferred from the public tensor layout | Medium |
| No residual add or normalisation inside the ht launches | Static reading (no residual read) | High |
| Bandwidth figures | Derived from public shapes and measured times | High for the arithmetic; DRAM traffic not counted |
| Batch-size columns | Whole-run medians, drafter-on steps only for 2 to 8 tokens | Medium (few launches for 6 tokens) |
| Output head path | Same single-matrix routine family (static reading). Token selection as two separate launches after it (trace) | High |
| Theoretical occupancy | Derived from register and shared-memory figures and public hardware limits | Medium (register-file size for gfx1151 assumed from public documentation) |
| Shared-expert gate (not ht) combination | Covered by the i4r spec (section 10): its scalar is computed at the end of the router launch and applied when the shared expert's output is added | High (from the earlier engagement) |

## Change log

- **1.0 (this document):** draft 3 as released, edited for publication.
  - Absolute timings and throughput are converted to relative figures (see "Numbers" above).
  - References to the i4r work point to the published i4r specification, where its addendum is now section 10.
  - No technical content is added beyond the released document.
