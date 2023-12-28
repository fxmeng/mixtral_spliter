# mixtral_spliter
Converting Mixtral-8x7B to Mixtral-[1~7]x7B 

The relation between MMLU accuracy and number of experts:

|  | -2 | -5 | -6 | -4 | -7 | -0 | -1 | -3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7x7b | 0.5937 | 0.5813 | 0.5873 | 0.5748 | 0.5790 | 0.5564 | 0.5179 | 0.0040 |
| 6x7b | — | 0.5448 | 0.5422 | 0.5417 | 0.5389 | 0.5359 | 0.4671 | 0.0296 |
| 5x7b | — | — | 0.4920 | 0.4827 | 0.4762 | 0.4674 | 0.3490 | 0.0004 |
| 4x7b | — | — | — | 0.4178 | 0.4138 | 0.3988 | 0.2918 | 0.0002 |
| 3x7b | — | — | — | — | 0.3553 | 0.3288 | 0.2723 | 0.2524 |
| 2x7b | — | — | — | — | — | 0.2760 | 0.2624 | 0.2510 |
| 1x7b | — | — | — | — | — | — | 0.2408 | 0.0028 |

For selecting expert 1 and expert 3, please use this command:
```
python select_experts.py  ---experts_ids 1,3 --source_dir $PATH/Mixtral-8x7B-v0.1 --output_dir $PATH/Mixtral-2x7B-v0.1
```
