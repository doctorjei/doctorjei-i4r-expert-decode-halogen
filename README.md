# Halogen i4r Expert (MoE) Decode Specification
> Version 1.0 (draft 1)

## Overview
This is a functional description of how halogen, Peonist LLC's inference engine, runs the routed-expert step of a mixture-of-experts model during decode, for weights in the i4r format: 4-bit values with one f16 scale per 128 values, in a 128-point Hadamard basis.

**Companion document:** [ht dense decode specification](ht-dense-decode.md) covers the engine's dense matrix-vector products (attention and linear-attention projections, output head, shared expert) for weights in the ht trellis format.

**Methods.** clean-room style; one actor studied the engine and wrote this description. A second actor reviewed to ensure no 'bleed over' into the clean specification. (The halogen license does not prohibit reverse engineering but does ask that derived source or kernels not be redistributed as one's own work; this document contains neither.)

**Numbers.** Absolute timings / throughput are deliberately omitted, per request in the halogen license (which asks that published figures name software version and prompt - and which are not relevant to this document). Out of respect for that request...
- times are given as shares of the per-token GPU busy time; and
- bandwidth is given as a share of the GPU's published peak memory bandwidth.

Other numbers (i.e., structural facts such as launch counts, work splits, register and memory use, and data sizes) are given exactly.

**Conditions.** Profiled runs used the model with its second-version weight set, 48 layers:
- gfx1151 GPU under a tracing profiler (adds small overhead per launch)
- two short prompts, 64 generated tokens each
  - 63 decode steps per request measured
  - first generated token comes from prompt pass
  - one request at a time with greedy sampling


## 1. Introduction

The routed-expert step of each layer, for each decode token, is run in _three (3) launches_ instead of 30; it reaches **65-72 percent** of GPU peak memory bandwidth on the expert weights, which is about seven (7) times that of other implementations for the format. This is accomplished via five (5) techniques:

1. **All selected experts in one launch per stage.** Routing runs in one launch, gate and up for all 10 experts in a second, and down for all 10 in a third. Expert selection stays on-device, read by expert launches directly, eliminating host round-trip.
2. **Finishing work is done by last arriving workgroup.** Gating non-linearity, Hadamard rotations, and weighted expert combination run in the workgroup finishing a tile last by using arrival counters instead of via separate launches / memory passes.
3. **Row-only split with wide contiguous loads issued early.** Each workgroup owns 64 output rows. Four lanes share and read a row front to back in 16-byte chunks. All of a row's loads are in flight before arithmetic, keeping memory busy.
4. **Cheap exact unpacking and packed arithmetic.** 4-bit codes become half-precision values with no integer conversion, multiplied in pairs against f16 input with 32-bit accumulation. Each 128-value block's scale is applied once to its block sum.
5. **Small, cheap rotations.** 128-point transforms are done in registers within one wavefront. Output-side transform is applied once to combined result rather than once per expert.


## 2. Launch Structure (per decode token)

| Stage | Per Layer | Per Token | Function |
|---|---|---|---|
| Router Projection&nbsp;&<br />Expert Selection | 1 | 48 | - Router projection for token<br />- Top-10 selection & normalized weights [3]<br />- Shared expert's gate/up projections<br />- Per-token gate value [10] |
| Expert Gate/Up Proj.&nbsp;& <br />Shared Expert Down Proj. | 1 | 48 | - Gate/up for selected experts<br />- Gating non-linearity<br />- Rotation into down proj.'s input basis<br />- Shared expert's activation/down projection <br />&nbsp;&nbsp;(Separate range of 100 workgroups) [10] |
| Expert Down Proj.&nbsp;&<br />Combine | 1 | 48 | - Down projection for selected experts<br />- Weighted sum over experts<br />- Output-side rotation<br />- Addition of Gated Shared-Expert Output <br />&nbsp;&nbsp;(Written once as layer's result) |
| Whole Token <br />(All Work) | ~13.4 | 644-668 | Everything - including attention, other mixers, <br />norms, & sampling |

### Additional Notes
- **Every layer** of 48 runs routed-expert step: 63 steps × 48 layers = 3,024 launches per three expert-step kinds per request
- **Fusion:** gate & up are fused into one launch; down is separate. Experts are batched: one launch per layer per stage, _not one per expert_.
- **Combine:** fused into down launch; no separate combine or scatter launch exists.
- **Shared expert** has no separate launch; its work is spread over launches above (section 10).
- **Handing off work inside one launch, without a second launch:**
  - workgroups write partial results (32b floats) to small scratch area, then increment per-tile arrival counter
  - workgroup arriving last for tile finishes the tile (non-linearity, rotation, combine) & resets counter
  - finishing work costs no extra launch & no global barrier
- **Graph capture:** none. Every decode launch is ordinary launch from host; no graph creation / launch calls.


## 3. Routing

- **Selection is produced & consumed on-device**, inside router launch:
  - router launch spreads router projection & other small per-token projections over many workgroups
  - last workgroup to finish runs selection: repeated arg-max rounds over router scores, using cross-lane exchanges within one wavefront, giving 10 highest-scoring experts
  - exponent-based normalization over selected scores
  - writes 10 expert indices & single-precision weights per token to device memory.
- **Two expert launches read indices directly.** Each workgroup derives its token, selection slot (0-9), & row range from workgroup number, then loads expert index for slot. Weight addresses are formed on-device from index.
- **No device-to-host copy or host synchronization between layers.**
  - Host issues all of token's launches (~650) back to back, synchronizing once per token, which matches reading back only the sampled token.
  - Trace holds 128 synchronizations, covering both requests: 64 per request, one for each generated token, including token from request's prompt pass.
- **Host launch-call time** totals about 6 percent of per-token GPU time and overlaps GPU work.
- **A short gap** of about 1 percent of per-token GPU time sits at start of each token, before first layer starts. It is host-side per-token work, not yet attributed.

## 4. Work Decomposition

**Model sizes used by these launches:**
- hidden size 2560;
- routed-expert intermediate size 640;
- 10 experts selected per token;
- 128-value scale blocks.

All three expert-step launches use wavefronts of 32 lanes & workgroups of 256 lanes (8 wavefronts).

| Stage | Workgroups per launch | Output rows per workgroup | Lanes per output row | Rows per wavefront | Workgroups per routed expert |
|---|---|---|---|---|---|
| Router and selection | 268 | not established | not established | not established | (not per expert) |
| Gate and up, routed part | 200 | 64 | 4 | 8 | 20 (10 for gate rows, 10 for up rows) |
| Gate and up, shared-expert range | 100 | see section 10 (tile-and-slice split) | | | (not per expert) |
| Down, all selected experts | 400 | 64 | 4 | 8 | 40 |

- **One expert's matrix is split across workgroups by output rows only.** Each workgroup takes 64 consecutive output rows and runs over the whole input length (2560 for gate and up, 640 for down). The input dimension is never split across workgroups, so no dot product needs a cross-workgroup reduction.
- **Gate and up of one expert form one stacked matrix** of 1280 rows (640 gate, 640 up): 10 × 1280 / 64 = 200 workgroups. Down: 10 × 2560 / 64 = 400 workgroups.
- **How experts share a launch:** each workgroup derives, from its workgroup number alone, the token, the selection slot (0 to 9) and its 64-row block within that expert's matrix. The workgroups for all 10 slots run in one launch, and the launch size doesn't depend on which experts were selected.
- **Within a workgroup**, four adjacent lanes share one output row, and each wavefront covers 8 rows. The four lanes split each scale block between them (section 5), and their four partial sums are added with cross-lane exchanges at the end of the row.
- **The split point** between the routed part and the shared-expert range of the gate-and-up launch is a launch argument. It was 200 in every captured launch.
- **Arrivals per 128-row tile** (the finishing step in section 2 runs when the last one arrives):
  - **gate and up:** 4 workgroups per tile of one selected expert, 2 holding the tile's 128 gate rows and 2 holding its 128 up rows, since the non-linearity needs both;
  - **down:** 20 workgroups per 128-row output tile, 2 per selected expert across all 10 experts, because the combine sums across experts.
- **Router launch:** its 268 workgroups compute the router projection, the shared expert's gate and up projections, and several other small per-token projections. The last workgroup to finish runs the selection (section 3). How the 268 workgroups are divided among those projections has not been decoded.

## 5. Memory access

- **Storage layout in device memory:** for each expert matrix, the 4-bit codes and the f16 scales are held in separate regions (codes for all rows, then scales), not as interleaved 66-byte blocks.
  - A row's codes are contiguous: 1280 bytes (20 blocks of 64 bytes) for gate and up, 320 bytes (5 blocks) for down.
  - Whether this layout is built at load time from the file's block layout was not examined.
- **Code loads:**
  - each lane loads 16 contiguous bytes (32 codes) from each 64-byte block of its row;
  - the four lanes of a row take the four 16-byte quarters, so together they read each block's 64 bytes in one go, and a row's codes are read front to back with full 16-byte (128-bit) loads;
  - adjacent rows are adjacent in memory, so a wavefront's 8 rows form one contiguous stretch.
- **Per lane per output row:** 320 bytes of codes for gate and up (20 × 16), 80 bytes for down (5 × 16).
- **Load issue:** the code loads for a row (or for a large group of blocks of a row) are issued together, before any arithmetic on them, so many 16-byte loads are in flight per lane. This is the main latency-hiding technique. There is no separate prefetch of the weights into LDS.
- **Scales:** read per row from the scale region with small vector loads, separately from the codes.
- **Activation (input) vector:**
  - It is already f16 when the launch starts and is copied into LDS without conversion. The conversion to f16 happens before the expert launches: for gate and up, together with the input-side rotation (section 7); for down, in the gate-and-up finishing step.
  - It is staged once per workgroup into LDS: 5 KiB of f16 for the 2560-value input of gate and up, 1.25 KiB for the 640-value input of down.
  - Each lane then reads it with 16-byte LDS reads matching its code quarters.
  - Weights never pass through LDS.
- **Intermediate results:**
  - one 32-bit float per output row per selected expert goes to a small scratch area for the finishing step (section 2);
  - the gate-and-up finishing step writes the 640-value f16 intermediate for each slot;
  - the down launch's finishing step writes the layer output once, in bf16.

## 6. Arithmetic

- **Unpacking:** nibbles are not converted through integers. Each 4-bit code is placed directly into the low mantissa bits of a half-precision number whose exponent is fixed so that the number equals a constant plus the code. One packed half-precision subtraction then gives the signed value (code minus 8) exactly, two values per operation.
- **Dot product:** packed half-precision pairs are multiplied with the matching pairs of the f16 input and accumulated into 32-bit floats by a two-way dot-product operation, so the values are never widened to 32 bits one by one.
- **Scale placement:** each 128-value block gets its own 32-bit accumulator. The block's f16 scale is applied once to that block sum (a single multiply-add into the row total), not per value.
- **Accumulation precision:** 32-bit floats throughout the dot product, the cross-lane reduction, the scratch partials and the weighted combine. Rounding to f16 happens only for the intermediate between gate-and-up and down, and rounding to bf16 only for the layer output.
- **No integer dot-product path** (8-bit or 4-bit) is used; the input stays in f16.

## 7. Rotation

| Where | What is applied | How often |
|---|---|---|
| Input of gate and up | Nothing inside the expert launches: the input vector arrives already in the rotated basis | Done before the router launch, once per layer and shared by all experts (the launch that does it was not identified) |
| Output of gate and up | 128-point Hadamard transform, 1/√128 scaling and the per-row sign vector, on the gate and the up results | Per selected expert, per 128-row tile, in the last-arriving workgroup of the gate-and-up launch |
| Input of down | Sign vector, then the 128-point Hadamard transform and 1/√128, applied after the gating non-linearity so the intermediate is already in down's input basis | Same place, same step, per selected expert |
| Output of down | 128-point Hadamard transform, 1/√128 and the sign vector | Once per layer, after the weighted sum of the 10 experts (valid because the transform is linear), in the last-arriving workgroup of the down launch |

- **All transforms are fused** into finishing steps of the matrix launches. There is no separate rotation launch.
- **The 128-point transforms** are fast Walsh–Hadamard transforms done in registers within one wavefront. Each lane holds 4 values of a 128-value tile, and the butterfly stages run partly inside a lane and partly by exchanging values between lanes. There is no LDS or memory round trip.
- **Cost:** the transforms run only in the finishing workgroups (one per 128-row tile), so they add a small tail to each launch rather than a pass over the data. They weren't measured separately; the estimate is a few percent of each launch's time at most.

## 8. Measurements (relative)

**How to read the table:**
- medians over 58 decode steps of one request (steps 6 to 63);
- **time** is the share of per-token GPU busy time;
- **bandwidth** is the share of the GPU's published peak memory bandwidth (256-bit LPDDR5X-8000);
- **weight bytes** are those read for the 10 selected experts (codes plus scales, 4.125 bits per value), with MB = 10^6 bytes.

| Kernel role (described, not named) | Share of per-token GPU time | Time per layer, relative to the down launch | Weight bytes per layer | Share of peak bandwidth | Registers per lane / LDS per workgroup | Waves per workgroup |
|---|---|---|---|---|---|---|
| Router and selection (including the shared expert's gate and up) | 6.7% | 0.74 | not established | not established | 96 vector registers, 14.5 KiB | 8 |
| Expert gate and up (fused, all experts), plus the shared expert's down | 19.8% | 2.20 | 16.90 MB routed, plus the shared expert's down matrix (format not established) | 65% (routed bytes only; a lower bound) | 112 vector registers, 10.5 KiB | 8 |
| Expert down (all experts) and combine | 9.1% | 1.00 | 8.45 MB | 72% | 96 vector registers, 1.5 KiB | 8 |
| Routed-expert step total | 35.6% | | 25.35 MB routed | 54% over all three launches; 67% over the two expert launches | | |
| Whole token, GPU busy | 100% | | | | | |
| Whole token, wall clock | about 107% of GPU busy time | | | | | |

- **The routed-expert step** takes about 36 percent of GPU time per token.
- **None of the three launches** spills registers to memory.
- **The shared expert's down matrix** is 2560 × 640. If it is stored at the same 4.125 bits per value, it adds about 0.85 MB per layer, and the gate-and-up launch would reach about 69 percent of peak. That is conditional, not measured.
- **The router launch's weight bytes** include the shared expert's gate and up matrices (1280 × 2560) as well as the router's own projection. Their format is not established, so the router launch's bandwidth stays open.
- **Comparison:** the open-source implementation of the same format reaches about 9 percent of peak, with about 30 launches per MoE layer (from our own earlier work, not measured in this analysis). The subject reaches about 65 to 72 percent with 3 launches.
- **Why it is fast, from the above:**
  - few launches, with all experts batched;
  - a row-only split with long runs of full-width contiguous loads issued ahead of use;
  - cheap exact nibble-to-f16 unpacking and packed dot products, with one scale multiply per block;
  - the activation staged once in LDS;
  - rotations and combine folded into the last-arriving workgroup instead of separate passes.
- **Test conditions:** see "About this document". All timings include a small per-launch tracing overhead, so the shares are slight overestimates for short launches.

## 9. Confidence

- **Router launch performs the selection:** high confidence. A last-arriving workgroup runs the top-10 selection and normalisation and writes indices and weights.
- **Gate and up fused with the gating non-linearity in one launch:** high confidence.
- **Combine inside the down launch,** with the output-side rotation applied once to the weighted sum, not once per expert: high confidence.
- **Work decomposition, memory access and arithmetic for the routed experts (sections 4 to 6):** high confidence. Read from the engine's compiled device code and matched to the captured launch sizes (200 and 400 routed workgroups, 64 rows each).
- **Shared expert (section 10):** high confidence for where each part runs, confirmed by a read-back test. Medium confidence on the exact tile shape of its down projection.
- **Input-side rotation (and conversion to f16) of the gate-and-up input:** done before the router launch, once per layer. High confidence that it is not inside the expert launches, since the routed path reads its input without any transform.
- **Cost of the in-launch rotations and finishing steps:** not measured. Estimated at a few percent of launch time at most (low-to-medium confidence, from the amount of work, not from timing).

**Still open:**
- the storage format and size of the shared expert's weights (both its gate and up, in the router launch, and its down, in the shared-expert range);
- how the router launch's 268 workgroups are divided among its projections;
- the exact tile shape of the shared-expert range (medium confidence, section 10);
- which launch does the input-side rotation before the router launch.

## 10. Shared expert

**Evidence:**
- static reading of the engine's compiled device code for the three expert-step launches;
- launch sizes and launch-time settings captured from a normal decode run;
- a read-back test:
  - at 20 sampled MoE layers in decode, spread over two requests, the engine was paused after each of the three expert-step launches;
  - at each pause, four small result vectors were summarised by whether each element was zero or not;
  - before the router launch, the vector that the down launch adds at its end was set to all zeros;
  - the test recorded which launch first wrote each vector;
  - all 20 samples completed, with none skipped, ended early or out of order, and the launch counts matched an uninstrumented run exactly.

| # | Finding | Confidence |
|---|---|---|
| 1 | The shared expert's gate and up projections (640 rows each, input 2560) run inside the router launch, as two extra grouped matrix-vector products beside the router's own projection. Both outputs are rewritten by the router launch in 20 of 20 samples, and by neither later launch. | High |
| 2 | The shared expert's per-token gate value (one value, passed through a sigmoid) is computed at the end of the router launch. It changes after the router launch in 20 of 20 samples, and not after the other two. | High |
| 3 | The shared expert's activation (SiLU of gate times up) and its down projection (2560 outputs from 640 inputs) run in the 100-workgroup shared-expert range of the gate-and-up launch, producing a 2560-value vector in the output basis. The zeroed vector stays all zeros after the router launch and is fully non-zero (2560 of 2560) after the gate-and-up launch in 20 of 20 samples. The down launch doesn't change it. | High |
| 4 | The down launch, in its finishing step, adds the gate value times that vector to the combined routed-expert output. So the shared expert adds no launch of its own. | High |
| 5 | The 100 workgroups come from splitting the shared down matrix (2560 × 640) into 20 output tiles of 128 rows and 5 input slices of 128 values, one tile-slice piece per workgroup. The intermediate is not wider than 640. | High that the split is along the input (captured settings show 640 inputs and a split factor of 5); medium on the exact tile shape (inferred from 100 = 20 × 5) |

**How the shared-expert range finishes a tile:** the five partial results for each tile are reduced by the last workgroup to arrive, the same pattern as for the routed experts. That workgroup then applies the output-side rotation and a single multiplier given with the launch settings before storing the tile. This multiplier is not the per-token gate value of finding 4, which is applied only in the down launch, and its purpose is not established.

**Data flow in one decode layer:**

| Launch (in order) | Shared-expert work done there | Reads | Writes |
|---|---|---|---|
| Router | The shared gate and up projections, beside the router projection and top-10 selection. Then the gate value (sigmoid). | Layer input (2560) | Shared gate (640), shared up (640), gate value (1) |
| Gate and up, shared-expert range (100 workgroups) | SiLU(gate) times up. Then the down projection as 20 tiles by 5 input slices, giving partial sums. The last arriver per tile sums, rotates and stores. | Shared gate and up (640 each) | Shared output vector (2560) |
| Down and combine | After the routed combine and output rotation, adds the gate value times the shared output vector. | Gate value, shared output vector | Layer output (2560) |

## Change log

- **1.0 (this document):** draft 1 and addendum 1 merged.
  - The addendum's shared-expert findings are folded into sections 2, 4, 8 and 9, and given in full in section 10.
  - Draft 1's medium-confidence reading of a five-times wider shared-expert intermediate, and the bandwidth figure derived from it, are withdrawn, as the addendum states.
  - Absolute timings and throughput are converted to relative figures (see "About this document").
  - The engine's single decode-rate figure is omitted.
  - Wording is edited for publication; no technical content is added beyond the two released documents.
