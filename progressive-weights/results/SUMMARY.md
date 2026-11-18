## 0. Reference points (WikiText-2 test, 8 windows x 512 tokens, Qwen2.5-0.5B)

| model | perplexity |
|---|---|
| original (bf16 weights, fp32 math) | 18.011 |
| 8-bit RTN, all tiles refined | 18.015 |
| base only, 4-bit (no refinement) | 21.406 |
| base only, 3-bit (no refinement) | 61.036 |
| base only, 2-bit (no refinement) | 89788.166 |

## 1. Quality vs. fraction of tiles refined per token

`f` = fraction of the input tiles that get their refinement bits for each token (tiles are 32 input channels wide). dynamic = chosen per token by |x_tile|*||R_tile||, static = fixed set chosen on calibration text (needs that extra RAM permanently), random = control. `read %` = refinement bytes touched per token as % of ALL refinement bytes (before any cache).

### base = 4 bit (RAM), refinement = 4 bit planes on SSD

| f | dynamic | static | random | read % (dynamic) | read MB/token |
|---|---|---|---|---|---|
| 0.05 | 19.73 | 20.50 | 21.27 | 4.1 | 7.3 |
| 0.10 | 19.27 | 19.83 | 20.99 | 10.5 | 18.8 |
| 0.20 | 19.08 | 19.35 | 20.72 | 21.0 | 37.5 |
| 0.40 | 18.56 | 18.85 | 19.81 | 39.6 | 70.9 |
| 0.70 | 18.22 | 18.30 | 18.65 | 71.1 | 127.2 |

### base = 3 bit (RAM), refinement = 5 bit planes on SSD

| f | dynamic | static | random | read % (dynamic) | read MB/token |
|---|---|---|---|---|---|
| 0.05 | 29.71 | 35.74 | 56.01 | 4.1 | 9.1 |
| 0.10 | 25.68 | 30.82 | 47.42 | 10.5 | 23.5 |
| 0.20 | 23.01 | 26.12 | 38.33 | 21.0 | 46.9 |
| 0.40 | 20.59 | 22.90 | 29.40 | 39.6 | 88.6 |
| 0.70 | 18.94 | 19.57 | 22.63 | 71.1 | 158.9 |

### base = 2 bit (RAM), refinement = 6 bit planes on SSD

| f | dynamic | static | random | read % (dynamic) | read MB/token |
|---|---|---|---|---|---|
| 0.05 | 6685.91 | 28242.94 | 58781.84 | 4.1 | 10.9 |
| 0.10 | 705.40 | 5728.38 | 33274.39 | 10.5 | 28.1 |
| 0.20 | 127.10 | 391.51 | 10321.39 | 21.0 | 56.3 |
| 0.40 | 39.78 | 79.38 | 1524.78 | 39.6 | 106.3 |
| 0.70 | 22.30 | 25.70 | 71.29 | 71.1 | 190.7 |

## 2. Coherence between consecutive tokens and energy coverage (dynamic mode)

overlap = fraction of a token's selected tiles that were also selected for the previous token (random selection would give about f). energy = share of sum(score^2) captured by the selected tiles.

| base | f | kind | overlap | energy covered |
|---|---|---|---|---|
| 4 | 0.05 | qkv | 0.53 | 0.20 |
| 4 | 0.05 | o | 0.37 | 0.17 |
| 4 | 0.05 | gu | 0.72 | 0.23 |
| 4 | 0.05 | down | 0.23 | 0.32 |
| 4 | 0.10 | qkv | 0.56 | 0.38 |
| 4 | 0.10 | o | 0.56 | 0.41 |
| 4 | 0.10 | gu | 0.68 | 0.38 |
| 4 | 0.10 | down | 0.29 | 0.42 |
| 4 | 0.20 | qkv | 0.61 | 0.55 |
| 4 | 0.20 | o | 0.65 | 0.62 |
| 4 | 0.20 | gu | 0.62 | 0.51 |
| 4 | 0.20 | down | 0.38 | 0.55 |
| 4 | 0.40 | qkv | 0.67 | 0.72 |
| 4 | 0.40 | o | 0.73 | 0.81 |
| 4 | 0.40 | gu | 0.63 | 0.66 |
| 4 | 0.40 | down | 0.54 | 0.73 |
| 4 | 0.70 | qkv | 0.82 | 0.91 |
| 4 | 0.70 | o | 0.86 | 0.96 |
| 4 | 0.70 | gu | 0.79 | 0.87 |
| 4 | 0.70 | down | 0.76 | 0.89 |
| 3 | 0.05 | qkv | 0.51 | 0.19 |
| 3 | 0.05 | o | 0.37 | 0.16 |
| 3 | 0.05 | gu | 0.71 | 0.22 |
| 3 | 0.05 | down | 0.23 | 0.31 |
| 3 | 0.10 | qkv | 0.56 | 0.37 |
| 3 | 0.10 | o | 0.56 | 0.39 |
| 3 | 0.10 | gu | 0.69 | 0.37 |
| 3 | 0.10 | down | 0.28 | 0.41 |
| 3 | 0.20 | qkv | 0.61 | 0.54 |
| 3 | 0.20 | o | 0.65 | 0.61 |
| 3 | 0.20 | gu | 0.62 | 0.50 |
| 3 | 0.20 | down | 0.38 | 0.54 |
| 3 | 0.40 | qkv | 0.67 | 0.72 |
| 3 | 0.40 | o | 0.73 | 0.81 |
| 3 | 0.40 | gu | 0.63 | 0.66 |
| 3 | 0.40 | down | 0.54 | 0.72 |
| 3 | 0.70 | qkv | 0.82 | 0.91 |
| 3 | 0.70 | o | 0.86 | 0.96 |
| 3 | 0.70 | gu | 0.78 | 0.87 |
| 3 | 0.70 | down | 0.76 | 0.89 |
| 2 | 0.05 | qkv | 0.52 | 0.16 |
| 2 | 0.05 | o | 0.35 | 0.14 |
| 2 | 0.05 | gu | 0.55 | 0.11 |
| 2 | 0.05 | down | 0.21 | 0.26 |
| 2 | 0.10 | qkv | 0.52 | 0.33 |
| 2 | 0.10 | o | 0.57 | 0.35 |
| 2 | 0.10 | gu | 0.64 | 0.29 |
| 2 | 0.10 | down | 0.28 | 0.38 |
| 2 | 0.20 | qkv | 0.60 | 0.52 |
| 2 | 0.20 | o | 0.64 | 0.57 |
| 2 | 0.20 | gu | 0.62 | 0.46 |
| 2 | 0.20 | down | 0.37 | 0.52 |
| 2 | 0.40 | qkv | 0.67 | 0.71 |
| 2 | 0.40 | o | 0.71 | 0.79 |
| 2 | 0.40 | gu | 0.62 | 0.65 |
| 2 | 0.40 | down | 0.53 | 0.71 |
| 2 | 0.70 | qkv | 0.82 | 0.90 |
| 2 | 0.70 | o | 0.86 | 0.95 |
| 2 | 0.70 | gu | 0.78 | 0.87 |
| 2 | 0.70 | down | 0.75 | 0.89 |

## 3. Never-wait policy: bounded fetch budget + resident-tile cache

budget = max NEW tiles fetched per token per group, as a fraction of the k wanted tiles (0 = never fetch, only reuse what is cached); cap = cache size in multiples of k (tiles kept resident). served = share of the wanted tiles that were usable.

base = 4 bit, f = 0.1

| cap | budget | perplexity | fetched MB/token | fetched % of refinement | served |
|---|---|---|---|---|---|
| 1.0 | 0.0 | 21.41 | 0.00 | 0.0 | 0.00 |
| 1.0 | 0.25 | 19.34 | 4.96 | 2.8 | 0.79 |
| 1.0 | 0.5 | 19.27 | 7.21 | 4.0 | 0.94 |
| 1.0 | inf | 19.27 | 8.32 | 4.6 | 1.00 |
| 2.0 | 0.0 | 21.41 | 0.00 | 0.0 | 0.00 |
| 2.0 | 0.25 | 19.32 | 3.23 | 1.8 | 0.88 |
| 2.0 | 0.5 | 19.26 | 4.66 | 2.6 | 0.97 |
| 2.0 | inf | 19.27 | 5.18 | 2.9 | 1.00 |
| inf | 0.0 | 21.41 | 0.00 | 0.0 | 0.00 |
| inf | 0.25 | 19.29 | 0.29 | 0.2 | 0.99 |
| inf | 0.5 | 19.27 | 0.30 | 0.2 | 1.00 |
| inf | inf | 19.27 | 0.30 | 0.2 | 1.00 |
