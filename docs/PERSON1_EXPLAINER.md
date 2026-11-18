# Person 1 explainer - what was built, why, and how to talk about it

(Use this to write the report sections in your own words and to prepare for the viva.
Every number here is from a real run; the script that produced it is named next to it.)

## 1. The problem as a machine-learning problem
GPS points are noisy, so we cannot just pick the nearest road. We turn map-matching into a
**ranking/classification problem**: for each GPS point, list the plausible road segments
(*candidates*), compute features for each (point, candidate) pair, and have a classifier score
"is this the road the vehicle is really on?". A classifier looks at one point at a time, so its
choices can be physically impossible (jumping between unconnected roads). An **HMM + Viterbi**
decoder then picks the best *sequence* of candidates using road connectivity.

## 2. Understanding the data (report section 2.2)
* One folder per trip: `.track` (lon, lat, time), `.nodes`, `.arcs` (directed edges between nodes, row index = arc id),
  `.route` (reference arc ids in order). Row i of the track does NOT correspond to row i of any network file -
  the route is a short arc sequence while the track has one row per second.
* Each trip has **its own network** and trips come from **different cities** (lon ~8.6, ~11.0, ~37.4). So:
  - distances must be in metres -> we project lon/lat to the **UTM zone of each trip** (`geo.py`). Degrees are
    angular, so Euclidean distance on them is wrong (1 deg lon is a different length at each latitude).
  - we do not use absolute coordinates as features (a model would memorise cities).
* Arcs are **directed**; ~82% of arcs have a reverse twin (two-way roads). Both directions have the same geometry, so
  distance alone cannot separate them -> **heading** is a key feature.
* Time: ~97% of steps are exactly 1 s; some trips have gaps up to 160 s (handled in the HMM, section 6).
* Partitions used: 80 train trips (everything here is fit/tuned only on them); 10 student-test trips for development
  comparison; evaluator data never touched.
* Sizes (`results` / `data/processed/*_trips.csv`): train 208,417 GPS points, networks of 299 to 155,003 nodes.

## 3. Candidate generation (`candidates.py`, `scripts/candidate_study.py`)
For each point: all arcs within R metres, nearest first, keep the K nearest (both directions of a road count
separately). A spatial index (STRtree) makes this fast on 300k-arc networks. If a point has none within R, R doubles
up to 400 m so every point has a candidate.
**Why R=50 m, K=20:** the study (`results/candidate_study.csv`) measures the *candidate miss rate* (reference arc not in
the candidate set): R=50, K=20 gives 0.06% misses with ~8 candidates/point; K=10 would give 0.29%, K=5 1.6%. Larger R adds
nothing because "valid" is defined as reference within 40 m (be honest about this circularity in the report).

## 4. Labels from the reference route (`labels.py`) - the subtle part
We only have a route (arc list), not "which arc for each GPS point". We align them with a **monotone dynamic programme**
(a Viterbi over route positions): each point gets a route index that never decreases, may stay or jump forward a few
arcs (vehicle speed), cost = distance from the point to that arc (capped at 60 m), small penalty per skipped arc.
A candidate is positive (`label=1`) if it equals the point's reference arc.
**Limitations to state in the report** (the guidelines ask for this):
* The reference is not guaranteed truth. Where GPS is far from the route (>40 m), the alignment is unreliable, so those
  points get `ref_valid=False` and are excluded from training/evaluation (11.6% of train points, 0.2% of student-test).
* Trips `00000054` and `00000033` have extremely noisy GPS (median 120-140 m from the reference route; 169 km of track
  for a 31 km route); `00000002` is 87% valid. Nearly all invalid points come from these trips.
* Labels are direction-aware: the reverse twin of the true arc is a *negative*.

## 5. Features (`features.py`, 28 features, all computed from the track + network only)
| group | features | why |
|---|---|---|
| geometry | `dist_m`, `lat_off_m`, `proj_frac`, `arc_len_m`, `along_m`, `to_end_m` | basic closeness; where along the arc the point projects |
| candidate-set context | `rank`, `dist_gap_to_best`, `dist_ratio_to_best`, `n_cand`, `has_reverse` | how this arc compares with its rivals; is it a two-way road |
| heading | `heading_diff`, `cos_heading`, `heading_valid` | direction of travel (from the points around i) vs arc direction - separates the two directions of a road and parallel streets. Unknown when stationary -> neutral 90 deg + flag |
| kinematics | `speed_mps`, `accel`, `turn_rate`, `dt_prev`, `dt_next` | driving behaviour and time gaps |
| temporal support | `d_prev1`, `d_next1`, `d_prev3`, `d_next3`, `support_mean` | distance of neighbouring GPS points (i+-1, i+-3) to the same arc: a real road is followed by neighbours, a noisy outlier is not |
| topology | in/out degree of the arc's start/end nodes | junction structure |
Impact: nearest-arc-only gets 56% top-1; adding a simple heading term (`dist + heading_diff/6`) gets 84.6% - heading
is the big win. A learned classifier on all 28 features (stand-in LightGBM) reaches 88.1% out-of-fold (89.2% student-test).
(Scaling/imputation: tree models need none; models that do must fit scalers on training folds only.)

## 6. HMM + Viterbi (`hmm.py`) - report section 5, answers to the 6 mandatory questions
1. **Hidden states:** at GPS observation i, the state is which candidate arc (<=20) the vehicle is on. The
   candidate<->arc correspondence is kept (`cand['arc']`, `pred_row`).
2. **Emission:** log of the classifier's probability for the candidate (floored at `eps` so one near-zero probability
   cannot veto a whole path). Per-point normalisation would not change the Viterbi path (constants cancel).
3. **Transition (Newson-Krumm):** network distance between the two positions vs straight-line distance between the
   two GPS points: `log P = -|d_net - d_gps| / beta`. `d_net = rest of arc a + shortest path(end(a) -> start(b)) + part
   of arc b` (bounded Dijkstra, memoised); same arc -> difference of positions. A vehicle moves about as far along the
   road as the GPS says, so large mismatches are penalised. Extra penalties for U-turns (b = reverse of a) and for
   moving backwards on one arc.
4. **Gaps / missing candidates:** `beta` grows with the time gap (capped), and the graph search radius grows with the
   jump length. Points with no candidate are skipped by the chain and take the last decoded arc.
5. **Disconnected consecutive observations:** unreachable pairs get a soft floor `-lam_break` instead of -inf, so the path
   can "restart" on a new part of the network instead of failing. `connect_route` then bridges non-adjacent consecutive
   arcs with the shortest path if one exists within 2.5 km, otherwise records a *route break*.
6. **Did it help?** (student_test, stand-in classifier, `results/hmm_eval_standin_lgbm_student_test.csv`):
   top-1 89.2% -> 94.1% with default parameters, 94.3% after tuning; route continuity 92.9% -> 98.1%; better on all 10
   trips. Parameter search (`scripts/tune_hmm.py`, 72 settings, train out-of-fold probabilities only): best
   beta=8, lam_break=24, lam_uturn=12, eps=1e-3. The best values sit at the edge of the grid, so the gain from
   tuning is small and a wider search might help slightly. Cases where it can fail: very noisy GPS where the emissions
   themselves are wrong for long stretches, and dense parallel two-way roads.
Viterbi recurrence: `delta_i(b) = logE_i(b) + max_a [delta_{i-1}(a) + logT(a->b)]`, with back-pointers to recover the path.
Everything expensive (shortest paths) is computed once per trip (`TransitionGeometry`), so tuning is cheap.
**Important for fairness:** decoder parameters are tuned on train (out-of-fold) + checked on student-test only; if the final
classifier changes, the decoder is re-tuned on that classifier's probabilities.

## 7. Likely viva questions (know your answers)
* Why UTM and not degrees / haversine? (metric, local, fast; zone per trip because trips are in different cities)
* Why group CV by trip? (points of one trip are highly correlated - splitting points leaks; the guidelines require it)
* Why are both directions separate candidates? (the route uses directed arc ids; heading disambiguates)
* How do you know your labels are right? (you don't fully: monotone alignment + validity flag; limitations stated)
* Why is the classifier not enough? (independent per-point decisions can jump between unconnected roads)
* What is the emission / transition? What happens on a GPS gap or disconnected graph? (section 6)
* How do you avoid leakage? (route read only for labels/scoring; features, candidates, decoder never use it; evaluator data untouched)
* What is the candidate miss rate and why does it matter? (upper bound on any ranker's top-1 accuracy)

## 8. Reproduce
```
python scripts/build_dataset.py            # tables
python scripts/candidate_study.py          # R/K choice
python scripts/dev_standin_probs.py        # stand-in classifier + OOF probs (decoder development only)
python scripts/tune_hmm.py --probs standin_lgbm
python scripts/eval_hmm.py --probs standin_lgbm --partition student_test
```
