# Tool-count benchmark — summary

Cell format: `TTFT_med / total_med ms` over response-started trials, plus the rate at which the model selected the relevant `get_weather` tool.

| Model | N=10 | N=50 | N=100 | N=200 | N=500 | N=1000 |
|---|---|---|---|---|---|---|
| `anthropic/claude-opus-4.7` | 1535 / 2360 ms<br>rel=100% | 2298 / 2918 ms<br>rel=100% | 1822 / 2798 ms<br>rel=100% | 3914 / 5219 ms<br>rel=100% | 2327 / 3157 ms<br>rel=100% | 5819 / 6595 ms<br>rel=100% |
| `anthropic/claude-sonnet-4.6` | 675 / 1822 ms<br>rel=100% | 1347 / 2046 ms<br>rel=100% | 1997 / 2739 ms<br>rel=100% | 1855 / 2576 ms<br>rel=100% | 2607 / 2613 ms<br>rel=100% | 2804 / 3224 ms<br>rel=100% |
| `deepseek-v4-flash` | 439 / 1439 ms<br>rel=100% | 652 / 1657 ms<br>rel=100% | 490 / 1513 ms<br>rel=100% | 635 / 1753 ms<br>rel=100% | 828 / 1776 ms<br>rel=100% | 1008 / 2184 ms<br>rel=100% |
| `deepseek-v4-pro` | 448 / 4124 ms<br>rel=100% | 655 / 4570 ms<br>rel=100% | 503 / 4447 ms<br>rel=100% | 651 / 4864 ms<br>rel=100% | 809 / 4791 ms<br>rel=67% | 1050 / 4995 ms<br>rel=100% |
