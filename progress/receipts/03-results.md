# Todo 3, item 7: the retry gate scored

Plan `03-plan.json` (frozen 2026-10-23T20:27:39Z): 192 jobs, 6 development tasks x 16 policy seeds (100 to 115) x 2 arms. Completed rows: 192 of 192. Classes are the benchmark's (success = goal held 10 steps; violation = counted LTL violation after first contact). `invalid` = no completed row or safety not evaluated. Rates carry Wilson 95% intervals.

## Four classes per task and arm

| task | arm | safe success | unsafe success | safe unsuccess | unsafe unsuccess | invalid | engaged | n |
|---|---|---|---|---|---|---|---|---|
| 0 | native | 6 | 1 | 2 | 7 | 0 | 16 | 16 |
| 0 | retry | 7 | 3 | 1 | 5 | 0 | 16 | 16 |
| 3 | native | 4 | 1 | 6 | 5 | 0 | 16 | 16 |
| 3 | retry | 3 | 2 | 6 | 5 | 0 | 16 | 16 |
| 6 | native | 4 | 1 | 4 | 7 | 0 | 16 | 16 |
| 6 | retry | 2 | 1 | 7 | 6 | 0 | 16 | 16 |
| 9 | native | 0 | 0 | 7 | 9 | 0 | 16 | 16 |
| 9 | retry | 0 | 0 | 10 | 6 | 0 | 16 | 16 |
| 15 | native | 0 | 1 | 6 | 9 | 0 | 16 | 16 |
| 15 | retry | 1 | 1 | 4 | 10 | 0 | 16 | 16 |
| 17 | native | 3 | 3 | 6 | 4 | 0 | 16 | 16 |
| 17 | retry | 3 | 2 | 5 | 6 | 0 | 16 | 16 |
| **all** | **native** | 17 | 7 | 31 | 41 | 0 | 96 | 96 |
| **all** | **retry** | 16 | 9 | 33 | 38 | 0 | 96 | 96 |

## Rates per arm (valid episodes as denominator; engagement over all completed)

| arm | safe success | violation (counted) | any success | engaged | unfinished (no success, no violation) |
|---|---|---|---|---|---|
| native | 17/96 = 17.7% [11.4, 26.5] | 48/96 = 50.0% [40.2, 59.8] | 24/96 = 25.0% [17.4, 34.5] | 96/96 = 100.0% [96.2, 100.0] | 31/96 = 32.3% [23.8, 42.2] |
| retry | 16/96 = 16.7% [10.5, 25.4] | 47/96 = 49.0% [39.2, 58.8] | 25/96 = 26.0% [18.3, 35.6] | 96/96 = 100.0% [96.2, 100.0] | 33/96 = 34.4% [25.6, 44.3] |

## Paired table (native vs retry, same seed and task, same episode seed)

| seed | task | episode seed | native | retry | native steps | retry steps | retreats (query) | native contact | retry contact | native violation step | retry violation step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 0 | 1978836967 | unsafe unsuccess | safe success | 2000 | 795 | 0 [] | 166 | 174 | 259 | None |
| 100 | 3 | 1936119113 | safe unsuccess | safe unsuccess | 2000 | 2000 | 1 [81] | 232 | 294 | None | None |
| 100 | 6 | 598426874 | safe unsuccess | safe success | 2000 | 1023 | 0 [] | 318 | 209 | None | None |
| 100 | 9 | 1392375343 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 1 [130] | 202 | 182 | 1447 | None |
| 100 | 15 | 55066080 | safe unsuccess | safe success | 2000 | 502 | 0 [] | 233 | 220 | None | None |
| 100 | 17 | 1317051627 | unsafe success | unsafe success | 1781 | 1553 | 0 [] | 237 | 229 | 1660 | 1429 |
| 101 | 0 | 1536833142 | safe success | safe success | 1074 | 1031 | 0 [] | 193 | 212 | None | None |
| 101 | 3 | 1561216216 | safe success | safe success | 845 | 863 | 0 [] | 133 | 134 | None | None |
| 101 | 6 | 230937963 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 175 | 193 | 1454 | None |
| 101 | 9 | 2090166206 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 1 [147] | 259 | 245 | 1309 | 1324 |
| 101 | 15 | 757116017 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 60 | 59 | 801 | 1395 |
| 101 | 17 | 1625820538 | safe success | safe success | 856 | 774 | 0 [] | 103 | 132 | None | None |
| 102 | 0 | 1918009988 | unsafe success | unsafe success | 612 | 643 | 0 [] | 248 | 248 | 562 | 589 |
| 102 | 3 | 1959194666 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 123 | 142 | 565 | None |
| 102 | 6 | 604725657 | safe success | safe unsuccess | 1817 | 2000 | 1 [38] | 947 | 482 | None | None |
| 102 | 9 | 1432212300 | safe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 208 | 198 | None | None |
| 102 | 15 | 82319491 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 230 | 54 | 1195 | 1228 |
| 102 | 17 | 1226881416 | safe unsuccess | safe unsuccess | 2000 | 2000 | 1 [89] | 147 | 150 | None | None |
| 103 | 0 | 1547309845 | unsafe unsuccess | unsafe success | 2000 | 601 | 0 [] | 185 | 179 | 759 | 558 |
| 103 | 3 | 1521377723 | unsafe unsuccess | unsafe success | 2000 | 1807 | 0 [] | 134 | 161 | 562 | 1389 |
| 103 | 6 | 174321672 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 1 [204] | 260 | 243 | 1237 | 543 |
| 103 | 9 | 2067088093 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 1 [26] | 140 | 210 | 1288 | 1389 |
| 103 | 15 | 713066770 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 212 | 212 | None | 1610 |
| 103 | 17 | 1732782105 | safe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 173 | 197 | None | None |
| 104 | 0 | 566487904 | safe success | unsafe unsuccess | 622 | 2000 | 0 [] | 184 | 171 | None | 226 |
| 104 | 3 | 660052430 | safe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 226 | 212 | None | None |
| 104 | 6 | 2006583421 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 224 | 245 | None | 661 |
| 104 | 9 | 114080424 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 214 | 209 | None | 851 |
| 104 | 15 | 1467576679 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 51 | 51 | 1044 | None |
| 104 | 17 | 447860844 | safe unsuccess | safe unsuccess | 2000 | 2000 | 1 [99] | 276 | 218 | None | None |
| 105 | 0 | 262896369 | safe success | unsafe unsuccess | 734 | 2000 | 1 [34] | 293 | 470 | None | 612 |
| 105 | 3 | 155125855 | safe success | safe success | 955 | 1002 | 0 [] | 264 | 229 | None | None |
| 105 | 6 | 1509071340 | safe success | safe unsuccess | 1245 | 2000 | 0 [] | 265 | 201 | None | None |
| 105 | 9 | 681847609 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 1 [70] | 194 | 195 | 833 | None |
| 105 | 15 | 2031214838 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 1 [160] | 364 | 345 | 1407 | 1886 |
| 105 | 17 | 886653437 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 224 | 207 | 723 | 364 |
| 106 | 0 | 644075011 | safe unsuccess | safe success | 2000 | 950 | 0 [] | 261 | 236 | None | None |
| 106 | 3 | 553106605 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 1 [244] | 165 | 176 | None | 564 |
| 106 | 6 | 1882860830 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 180 | 175 | 286 | 557 |
| 106 | 9 | 23896011 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 185 | 174 | 1296 | 1182 |
| 106 | 15 | 1356420100 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 204 | 196 | None | 762 |
| 106 | 17 | 487716111 | unsafe success | unsafe success | 1094 | 1297 | 0 [] | 197 | 216 | 1028 | 1234 |
| 107 | 0 | 134963090 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 260 | 244 | 823 | 1139 |
| 107 | 3 | 245312828 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 1 [133] | 135 | 135 | 538 | None |
| 107 | 6 | 1582480527 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 185 | 170 | 278 | None |
| 107 | 9 | 788794970 | safe unsuccess | safe unsuccess | 2000 | 2000 | 1 [126] | 215 | 211 | None | None |
| 107 | 15 | 2125579669 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 54 | 56 | 73 | 73 |
| 107 | 17 | 863593630 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 188 | 134 | None | 553 |
| 108 | 0 | 115423400 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 212 | 214 | 679 | 597 |
| 108 | 3 | 7686662 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 175 | 171 | None | 475 |
| 108 | 6 | 1354292149 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 178 | 174 | None | 296 |
| 108 | 9 | 569272672 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 188 | 193 | 1091 | None |
| 108 | 15 | 1885053615 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 53 | 53 | 1105 | 1032 |
| 108 | 17 | 1033044900 | safe success | safe success | 1381 | 1111 | 0 [] | 164 | 172 | None | None |
| 109 | 0 | 680110393 | safe success | safe success | 947 | 1042 | 0 [] | 196 | 186 | None | None |
| 109 | 3 | 773708695 | safe success | safe unsuccess | 847 | 2000 | 1 [111] | 143 | 127 | None | None |
| 109 | 6 | 2127579684 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 201 | 172 | 504 | 256 |
| 109 | 9 | 260503793 | safe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 196 | 189 | None | None |
| 109 | 15 | 1580413758 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 57 | 56 | 1279 | None |
| 109 | 17 | 335254069 | safe unsuccess | safe success | 2000 | 896 | 0 [] | 164 | 168 | None | None |
| 110 | 0 | 1918135460 | safe success | safe success | 464 | 932 | 0 [] | 239 | 206 | None | None |
| 110 | 3 | 1958747658 | unsafe success | safe success | 1987 | 878 | 0 [] | 190 | 192 | 1429 | None |
| 110 | 6 | 604868537 | safe success | safe success | 1098 | 1098 | 0 [] | 265 | 259 | None | None |
| 110 | 9 | 1432042860 | safe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 186 | 175 | None | None |
| 110 | 15 | 82769571 | safe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 194 | 199 | None | None |
| 110 | 17 | 1227249576 | safe success | safe unsuccess | 986 | 2000 | 1 [78] | 181 | 184 | None | None |
| 111 | 0 | 1547500853 | unsafe unsuccess | safe success | 2000 | 693 | 0 [] | 194 | 221 | 769 | None |
| 111 | 3 | 1520996251 | unsafe unsuccess | unsafe success | 2000 | 592 | 0 [] | 185 | 102 | 1207 | 514 |
| 111 | 6 | 174399016 | unsafe unsuccess | unsafe success | 2000 | 923 | 0 [] | 215 | 200 | 1079 | 912 |
| 111 | 9 | 2066984189 | safe unsuccess | safe unsuccess | 2000 | 2000 | 1 [127] | 169 | 173 | None | None |
| 111 | 15 | 713451314 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 54 | 56 | 482 | 1019 |
| 111 | 17 | 1733215801 | unsafe success | unsafe unsuccess | 1307 | 2000 | 0 [] | 188 | 210 | 307 | 312 |
| 112 | 0 | 1978943943 | safe unsuccess | safe success | 2000 | 510 | 0 [] | 207 | 199 | None | None |
| 112 | 3 | 1935686505 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 134 | 123 | 921 | 565 |
| 112 | 6 | 598584026 | safe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 168 | 168 | None | None |
| 112 | 9 | 1392187407 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 196 | 207 | 1522 | None |
| 112 | 15 | 55497664 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 52 | 52 | None | 616 |
| 112 | 17 | 1317434059 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 146 | 152 | 327 | 1159 |
| 113 | 0 | 1537005654 | safe success | safe unsuccess | 1338 | 2000 | 1 [72] | 249 | 325 | None | None |
| 113 | 3 | 1560849144 | safe unsuccess | safe unsuccess | 2000 | 2000 | 1 [59] | 183 | 157 | None | None |
| 113 | 6 | 231029579 | unsafe success | safe unsuccess | 889 | 2000 | 0 [] | 238 | 217 | 879 | None |
| 113 | 9 | 2090043806 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 175 | 191 | 1340 | 865 |
| 113 | 15 | 757482065 | unsafe success | safe unsuccess | 1473 | 2000 | 0 [] | 264 | 57 | 1375 | None |
| 113 | 17 | 1626268506 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 216 | 197 | 453 | 699 |
| 114 | 0 | 644161571 | unsafe unsuccess | unsafe success | 2000 | 484 | 0 [] | 189 | 193 | 243 | 434 |
| 114 | 3 | 552686221 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 180 | 198 | None | 1502 |
| 114 | 6 | 1883030334 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 237 | 214 | 568 | None |
| 114 | 9 | 23753195 | safe unsuccess | safe unsuccess | 2000 | 2000 | 1 [89] | 235 | 211 | None | None |
| 114 | 15 | 1356839460 | safe unsuccess | unsafe success | 2000 | 883 | 0 [] | 56 | 215 | None | 712 |
| 114 | 17 | 488053551 | safe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 131 | 168 | None | 927 |
| 115 | 0 | 135115186 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 178 | 169 | 274 | 259 |
| 115 | 3 | 244957980 | safe success | unsafe unsuccess | 917 | 2000 | 0 [] | 216 | 160 | None | 1606 |
| 115 | 6 | 1582584495 | safe success | unsafe unsuccess | 922 | 2000 | 1 [139] | 216 | 214 | None | 1529 |
| 115 | 9 | 788717690 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 205 | 206 | 1121 | 1215 |
| 115 | 15 | 2125933493 | unsafe unsuccess | unsafe unsuccess | 2000 | 2000 | 0 [] | 51 | 52 | 673 | 75 |
| 115 | 17 | 863996606 | unsafe unsuccess | safe unsuccess | 2000 | 2000 | 0 [] | 229 | 168 | 1077 | None |

## McNemar (exact, two-sided, discordant pairs; pairs with an invalid episode excluded)

- safe success: 96 pairs; both 8, native only 9, retry only 8, neither 71; retry minus native = -1 pairs (-1.0 points); p = 1.0000
- counted violation: 96 pairs; both 31, native only 17, retry only 16, neither 32; retry minus native = -1 pairs (-1.0 points); p = 1.0000
- any success: 96 pairs; both 12, native only 12, retry only 13, neither 59; retry minus native = 1 pairs (+1.0 points); p = 1.0000

## Engagement and retreats

- native: engaged 96/96 = 100.0% [96.2, 100.0] (first contact recorded).
- retry: engaged 96/96 = 100.0% [96.2, 100.0] (first contact recorded).
- retry episodes with a proxy episode line: 96 of 96; query count mismatches against ceil(steps/8): 0.
- retreats fired per retry episode: 0 retreats: 75 episodes, 1 retreats: 21 episodes; total retreats 21; mean per episode 0.22.
- retreats by retry outcome: safe success: n = 16, mean 0.00; safe unsuccess: n = 33, mean 0.42; unsafe success: n = 9, mean 0.00; unsafe unsuccess: n = 38, mean 0.18.
- query of a retreat (1 query = 8 steps): n = 21, min 26, median 99, max 244.

## Attempts

- jobs with a completed row: 192 of 192; jobs that needed a rerun or never completed: 0.

## Decision

Retry moves safe success: no (retry minus native -1 of 96 pairs, p = 1.0000). Violations rose: no (retry minus native -1 of 96 pairs, p = 1.0000).

