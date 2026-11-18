# Dataset exploration and assumptions

The raw audit covers the supplied `train` and `student_test` directories. The evaluator partition was outside the audit. `files_by_trip.csv` records availability, byte sizes, parsed dimensions, and field names for all expected files. `trips.csv` contains per-trip observations, extent, timing, network, and route statistics. The two summary CSVs aggregate the audit by partition and by file type.

| Partition | Trips | GPS observations | Expected files present | Exactly 1 s | Largest gap | Networks with multiple weak components |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 80 | 208,417 | 320 / 320 | 96.50% | 153,064 s | 79 / 80 |
| `student_test` | 10 | 19,160 | 40 / 40 | 98.62% | 210 s | 10 / 10 |

- All 360 files parsed with their expected column counts. The two partitions had zero nonfinite GPS rows, zero out-of-range route IDs, and zero isolated nodes. Multiple components can occur without isolated nodes when groups of junctions have roads within each group.
- The longest consecutive timestamp gaps occurred in `train/00000090` and `student_test/00000065`. Train had 590 intervals longer than 10 seconds, including four longer than 1,000 seconds. Student test had 27 longer than 10 seconds. The percentage at 1 s uses all finite intervals within each partition as the denominator.
- GPS coordinate plots use degrees. Physical extent and distance use a local UTM projection per trip. `gps_path_km` sums straight-line distances between successive plausible observations, including across long timestamp gaps. GPS noise and gaps can make that estimate differ from the driven road distance.
- Weak connectivity ignores direction. A road network in one weak component does not guarantee that every directed turn or route is drivable. The largest-component fraction is in `trips.csv` alongside the number of components.
- A `.route` row stores a zero-based arc ID in travel order. The raw audit found 13 consecutive reference pairs in train and one in student test whose endpoints did not join directly. The processed reference-alignment fields also flag low coverage on train trips `00000054` and `00000033`. Use the supplied route as a reference with these limitations in mind.
- The trajectory figure shows a trip close to each partition's median observation count and the trip with its largest gap. The model-free Folium maps use the near-median trips and show GPS observations and nearby road segments. Keep the HTML under `data/processed/eda_maps/` because it embeds raw coordinates.

Open `notebooks/dataset_exploration.ipynb` to reproduce the audit and inspect its outputs, or run `notebooks/dataset_exploration.py` with `PATHSNAP_DATA` set. Codex helped write and run the audit. The tables and plots were generated from the local student files for team review.
