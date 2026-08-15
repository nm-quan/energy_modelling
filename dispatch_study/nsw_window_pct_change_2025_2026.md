# NSW1 11:00-14:00 — May → July percentage change, 2025 vs 2026

Daily average over the 36 five-minute intervals of the window, then averaged across the days of each month. Source: OpenElectricity v4, NSW1, 5-minute.

| series | 2025 May | 2025 Jul | **2025 %** | 2026 May | 2026 Jul | **2026 %** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| demand | 6,579.3 | 7,410.5 | **+12.6%** | 6,734.5 | 7,136.9 | **+6.0%** |
| net demand | 3,734.1 | 4,147.7 | **+11.1%** | 3,857.4 | 3,715.1 | **-3.7%** |
| price $/MWh | 36.3 | 38.2 | **+5.2%** | 60.0 | 52.1 | **-13.2%** |
| |  |  |  |  |  |  |
| coal (black) | 3,661.0 | 4,024.3 | **+9.9%** | 3,946.6 | 4,120.6 | **+4.4%** |
| **gas total** | 71.3 | 59.1 | **-17.1%** | 62.3 | 42.6 | **-31.7%** |
|   gas CCGT | 32.2 | 13.6 | **-57.6%** | 9.8 | 0.0 | **-100.0%** |
|   gas OCGT | 37.8 | 45.5 | **+20.4%** | 52.2 | 42.6 | **-18.4%** |
|   distillate | 1.4 | 0.0 | **-100.0%** | 0.4 | 0.0 | **-97.8%** |
| hydro | 72.5 | 5.0 | **-93.0%** | 89.5 | 15.5 | **-82.7%** |
| battery out | 6.5 | 40.3 | **+515.7%** | 49.2 | 42.5 | **-13.5%** |
| battery in | 117.6 | 160.7 | **+36.6%** | 607.5 | 1,004.1 | **+65.3%** |
| |  |  |  |  |  |  |
| wind | 664.5 | 1,048.2 | **+57.7%** | 625.8 | 750.5 | **+19.9%** |
| solar utility | 2,180.7 | 2,214.5 | **+1.5%** | 2,251.2 | 2,671.3 | **+18.7%** |
| solar rooftop | 2,556.3 | 2,711.9 | **+6.1%** | 2,867.2 | 3,435.9 | **+19.8%** |
| **wind + grid solar** | 2,845.3 | 3,262.8 | **+14.7%** | 2,877.1 | 3,421.8 | **+18.9%** |
| bioenergy | 8.7 | 16.8 | **+93.9%** | 6.3 | 20.4 | **+222.7%** |
| pumps | 334.8 | 352.7 | **+5.3%** | 78.1 | 90.9 | **+16.3%** |

## Gas is not missing — it is an evening peaker

The window is midday, when solar has displaced it. Gas here is `gas_ccgt + gas_ocgt + distillate`; NSW1 has no `gas_steam` at all (VIC1 does), so this is the whole gas fleet.

| gas total (MW) | 2025 | 2026 |
| --- | ---: | ---: |
| mean 11:00-14:00 | 66 | 91 |
| mean 17:00-20:00 | 959 | 316 |
| mean, all hours | 301 | 149 |
| max 5-minute | 2,322 | 1,402 |

_Evening gas fell from 959 MW to 316 MW between the two years — a bigger structural change than anything in the midday window._

