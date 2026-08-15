# NSW1 — the 11:00-14:00 window, Jul 1 – Aug 1 2025 vs May 1 – Jun 1 2025

Daily average over the 36 five-minute intervals from 11:00 to 14:00, then averaged across the days of each month. May 1 – Jun 1 2025: 31 days. Jul 1 – Aug 1 2025: 31 days. Source: OpenElectricity v4, NSW1, 5-minute.

`demand` is operational demand and is **already net of rooftop solar** — verified on this data: it falls to 6,955 MW at noon versus 8,022 MW at midnight while rooftop peaks at 2,801 MW. `net demand` subtracts grid-scale wind and solar on top of that.

## The headline

| series | May 1 – Jun 1 2025 | Jul 1 – Aug 1 2025 | change | % |
| --- | ---: | ---: | ---: | ---: |
| demand (MW) | 6,579 | 7,410 | +831 | +12.6% |
| net demand (MW) | 3,734 | 4,148 | +414 | +11.1% |
| price ($/MWh) | 36 | 38 | +2 | +5.2% |

_Wind + grid solar in the window went 2,845 -> 3,263 MW, which is why demand and net demand move by different amounts._

## Spread across days (not just the mean)

| series | period | mean | sd | min | median | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| demand | May 1 – Jun 1 2025 | 6,579.3 | 1,012.3 | 4,791.9 | 6,422.6 | 8,844.1 |
| demand | Jul 1 – Aug 1 2025 | 7,410.5 | 1,299.5 | 5,581.5 | 6,967.7 | 11,035.1 |
| net demand | May 1 – Jun 1 2025 | 3,734.1 | 1,244.7 | 2,037.4 | 3,391.9 | 6,414.9 |
| net demand | Jul 1 – Aug 1 2025 | 4,147.7 | 1,504.0 | 1,915.0 | 3,748.1 | 7,506.5 |
| price | May 1 – Jun 1 2025 | 36.3 | 41.4 | -27.1 | 38.9 | 105.5 |
| price | Jul 1 – Aug 1 2025 | 38.2 | 48.6 | -14.6 | 28.4 | 229.8 |

## Fuel mix in the window (mean MW)

| fuel | May 1 – Jun 1 2025 | Jul 1 – Aug 1 2025 | change | % |
| --- | ---: | ---: | ---: | ---: |
| coal_black | 3,661.0 | 4,024.3 | +363.4 | +9.9% |
| gas_ccgt | 32.2 | 13.6 | -18.5 | -57.6% |
| gas_ocgt | 37.8 | 45.5 | +7.7 | +20.4% |
| hydro | 72.5 | 5.0 | -67.5 | -93.0% |
| battery_discharging | 6.5 | 40.3 | +33.8 | +515.7% |
| battery_charging | 117.6 | 160.7 | +43.0 | +36.6% |
| wind | 664.5 | 1,048.2 | +383.7 | +57.7% |
| solar_utility | 2,180.7 | 2,214.5 | +33.8 | +1.5% |
| solar_rooftop | 2,556.3 | 2,711.9 | +155.7 | +6.1% |
| bioenergy_biomass | 8.7 | 16.8 | +8.1 | +93.9% |
| distillate | 1.4 | 0.0 | -1.4 | -100.0% |
| pumps | 334.8 | 352.7 | +17.9 | +5.3% |

