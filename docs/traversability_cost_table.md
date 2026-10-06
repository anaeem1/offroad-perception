# RELLIS-3D class → traversability cost (v0.1)

Source of truth: [`config/class_costs.yaml`](../config/class_costs.yaml). This page explains it.

**Scale:** `0` free … `254` lethal, `255` UNKNOWN (never free, per the contract).
**Assumed machine:** wheeled, Warthog-class (the RELLIS-3D platform). A tracked construction machine
would lower mud and rubble. That is a per-platform config change, not a code change.

| ID | Class | Cost | Tracked obstacle | Rationale |
|---|---|---|---|---|
| 10 | asphalt | 0 | | Paved |
| 23 | concrete | 0 | | Paved |
| 1 | dirt | 10 | | Natural ground, firm |
| 3 | grass | 40 | | Drivable, but hides stumps and holes |
| 31 | puddle | 80 | | Shallow by definition; can hide ruts |
| 33 | mud | 140 | | Immobilization risk |
| 34 | rubble | 160 | | Loose, irregular; slow speed only |
| 19 | bush | 180 | | Possibly pushable, but blocks the view behind it |
| 4 | tree | 254 | | Static structure: grid only |
| 12 | building | 254 | | Static structure: grid only |
| 18 | fence | 254 | | Static structure: grid only |
| 6 | water | 254 | | Deep water; depth is not observable |
| 5 | pole | 254 | yes | Thin enough to be under-represented in a 0.2 m grid |
| 8 | vehicle | 254 | yes | Movable |
| 9 | object | 254 | yes | Movable / discrete |
| 15 | log | 254 | yes | Discrete; can be moved |
| 17 | person | 254 | yes | Safety-critical; consumers inflate margins |
| 27 | barrier | 254 | yes | Discrete, often repositioned on sites |
| 0 | void | 255 | | Unlabeled / unsure → UNKNOWN |
| 7 | sky | — | | No ground cell; a lidar point labeled sky is a misclassification and is dropped |

**Tracked obstacle rule:** a class is clustered and tracked if it is discrete and either movable or too
thin for the grid. Extended static structure (trees, buildings, fences, water) appears only as lethal
grid cells, which keeps the tracker from flooding in forested scenes.

**Rules the BEV stage must follow:**
1. Semantic cost is a prior. Geometric checks (step height, slope) may raise a cell's cost, never lower it.
2. A cell with no observations stays `255`. Absence of evidence is not free space.
3. When classes disagree within a cell, the cell takes the highest cost that has enough support
   (threshold set in week 2 and recorded with its measured effect on IoU).

**Open (week 2):** costs are engineering judgment, not measured. They get revisited once the BEV grid
runs on the replayed bag; any change is logged here with the reason.
