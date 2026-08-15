# NSW1 — the 11:00-14:00 window, Jul 1 – Aug 1 vs May 1 – Jun 1

Daily average over the 36 five-minute intervals from 11:00 to 14:00, then averaged across the days of each month. May 1 – Jun 1: 31 days. Jul 1 – Aug 1: 31 days. Source: OpenElectricity v4, NSW1, 5-minute.

`demand` is operational demand and is **already net of rooftop solar** (verified: it falls to 7,020 MW at noon while rooftop peaks at 3,152 MW). `net demand` subtracts grid-scale wind and solar on top of that.

## The headline

| series | May 1 – Jun 1 | Jul 1 – Aug 1 | change | % |
| --- | ---: | ---: | ---: | ---: |
| demand (MW) | 6,734 | 7,137 | +402 | +6.0% |
| net demand (MW) | 3,857 | 3,715 | -142 | -3.7% |
| price ($/MWh) | 60 | 52 | -8 | -13.2% |

_Wind + grid solar in the window went 2,877 -> 3,422 MW, which is why demand and net demand move by different amounts._

## Spread across days (not just the mean)

| series | period | mean | sd | min | median | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| demand | May 1 – Jun 1 | 6,734.5 | 1,010.4 | 4,891.5 | 6,836.6 | 8,715.8 |
| demand | Jul 1 – Aug 1 | 7,136.9 | 703.9 | 5,989.0 | 6,926.1 | 8,617.0 |
| net demand | May 1 – Jun 1 | 3,857.4 | 1,586.3 | 1,239.0 | 3,458.3 | 7,719.4 |
| net demand | Jul 1 – Aug 1 | 3,715.1 | 1,056.7 | 1,479.3 | 3,500.9 | 5,384.3 |
| price | May 1 – Jun 1 | 60.0 | 57.6 | -0.2 | 47.4 | 276.9 |
| price | Jul 1 – Aug 1 | 52.1 | 24.3 | -3.0 | 53.7 | 90.1 |

## Fuel mix in the window (mean MW)

| fuel | May 1 – Jun 1 | Jul 1 – Aug 1 | change | % |
| --- | ---: | ---: | ---: | ---: |
| coal_black | 3,946.6 | 4,120.6 | +173.9 | +4.4% |
| gas_ccgt | 9.8 | 0.0 | -9.8 | -100.0% |
| gas_ocgt | 52.2 | 42.6 | -9.6 | -18.4% |
| hydro | 89.5 | 15.5 | -74.0 | -82.7% |
| battery_discharging | 49.2 | 42.5 | -6.6 | -13.5% |
| battery_charging | 607.5 | 1,004.1 | +396.6 | +65.3% |
| wind | 625.8 | 750.5 | +124.7 | +19.9% |
| solar_utility | 2,251.2 | 2,671.3 | +420.1 | +18.7% |
| solar_rooftop | 2,867.2 | 3,435.9 | +568.6 | +19.8% |
| bioenergy_biomass | 6.3 | 20.4 | +14.1 | +222.7% |
| pumps | 78.1 | 90.9 | +12.8 | +16.3% |

