# How the no-model fill works

Plain description of `imputation/nem_baseline.py`. No neural network is involved.
Every number in the rule is measured somewhere in `dispatch_study/`, and the file
says where.

---

## The problem

A 3-hour window (11:00–14:00) is blanked out. Inside it we do not know:

- the 6 dispatch channels — hydro, coal, gas steam, gas OCGT, battery charging, battery discharging
- the 2 curtailment channels — wind curtailment, solar curtailment

We do know, at every 5-minute step:

- **net demand** — what the six fuels must add up to
- **the two edges** — the real dispatch at 10:55 and at 14:00, and the real curtailment at those same two instants
- **price**
- **the physical limits** — each channel's cap and its per-5-minute ramp limit, from `ml/check_caps.py`

And the counterfactual says: demand goes **up 2.4% inside the window** and **down 2.4% everywhere else**.

---

## Step 0 — the rest of the day

Outside the window, every channel is multiplied by `nd_after / nd_before`. This is
planA's "scaled" mode and applies the 2.4% reduction on both sides of the window.

Multiplying every channel by the same number keeps the balance equation true
automatically, so the off-window needs no repair.

The two edge values `pL` and `pR` are then taken from this **scaled** day, not the
original, so the window joins up with the world around it.

---

## Step 1 — draw a straight line

For all 8 channels, draw a straight line from the left edge to the right edge.

```
value at step t  =  left_edge  +  (t / 37) × (right_edge − left_edge)
```

That is the "nothing surprising happened" guess. It is the starting point, not the
answer. It is also, on its own, the most accurate thing anyone in this project has
produced — plain interpolation scores 50.6 MW MAE on the dispatch channels.

---

## Step 2 — decide how much curtailment gets used

This is the part that answers "does curtailment go to zero". It does not.

Curtailment gets split by **price**, because price tells you *why* the energy was
spilled:

| price at that step | what it means | what we do |
|---|---|---|
| **below zero** | The generator stopped because running was losing it money. Economic. | **Release it.** Higher demand lifts the price, so it would run. |
| **zero or above** | Price was fine and it *still* got spilled. That is a transmission or system-strength limit. | **Keep it.** More load in Melbourne does not build a new powerline out of western Victoria. |

Measured over 2022–2026 across 7,681,136 MWh of curtailment: **84.7% sat below zero
price, 15.3% did not**. And 85.4% of all curtailment happens in the lowest
net-demand quintile — when demand is already high, there is barely any left to
release.

One extra limit: you cannot deliver more free energy than there is new demand to
absorb it. So at each step the release is capped at that step's demand increase.

```
released(t) = min( curtailment(t) if price(t) < 0 else 0 ,  demand increase(t) )
```

Over the 10 days this releases **4,511 MWh of 12,546** — the other **64% stays spilled**.

---

## Step 3 — work out what the fuels still have to cover

```
shortfall(t) = net demand(t) − released curtailment(t) − (the straight line's total)
```

The released renewables count as supply, so they reduce what the fuels must produce.
That is the whole mechanism by which "free electricity" shows up in the answer.

---

## Step 4 — share the shortfall out

Not evenly, and not by a weight I chose. By the **measured** response of the real
grid over a 3-hour horizon (`dispatch_study/response.md`, Study A). Those numbers
come from regressing every fuel's 3-hour change against the 3-hour change in net
demand, across 474,300 pairs:

| channel | MW per MW of net-demand change |
|---|---|
| coal | 0.336 |
| hydro | 0.227 |
| gas OCGT | 0.075 |
| battery discharging | 0.057 |
| battery charging | −0.051 (charges *less*) |
| gas steam | 0.029 |

So coal takes the largest share, hydro next, gas steam barely moves. That is what
the grid actually does, not an assumption.

---

## Step 5 — respect the physical limits, step by step

Working forwards through the 36 steps, each channel is only allowed to move within:

- its **ramp limit** from where it actually ended up on the previous step
- its **cap**, and not below zero
- on the last step only, close enough to `pR` that it can join the right edge

If a channel hits a limit, its unused share is handed to whatever still has room,
and the sharing repeats up to 4 times. That is why the totals still add up exactly
even when something maxes out.

---

## What this guarantees

Over 10 days × 36 steps × 6 channels, measured:

```
balance errors above 1 MW : 0
ramp limit breaches       : 0
negative generation       : 0
unserved energy           : 0 MWh
```

And the accounting closes exactly:

```
fuels supply     +33,047 MWh
released renewables  +4,511 MWh
                 ───────────
total            +37,558 MWh   =  the demand increase, to the MWh
```

---

## Where it is wrong

**Coal 33.0% against a measured 43.4%, battery charging 15.4% against 6.6%.**

The rule takes the shortfall out of midday battery charging too aggressively.
Charging is at its daily maximum at 11:00–14:00 (155 MW average), so it looks like
cheap headroom. The real grid protects that charging because it has an evening
discharge to serve — the battery is storing solar for the 18:00 peak.

The rule has no concept of state of charge, so it cannot see that obligation. This
is exactly the kind of cross-time coupling a learned model ought to discover, and
so far neither trained model has.

**The right-edge check only runs on the final step.** A proper version would apply
the backward-reachability cone at every step (as `recursive_head.py` does), so the
trajectory can never paint itself into a corner. Here it does not matter — unserved
energy came out at 0 — but on a window with a larger jump between the two edges it
could.

**The price split is coarse.** Treating every below-zero step as fully economic
over-releases at those steps, and treating every above-zero step as fully
network-bound under-releases at those. The two errors offset to the measured 84.7%
on aggregate, but not step by step.

---

## Why this baseline matters

It reproduces four of six channels to within 0.4 percentage points of the real
grid's 3-hour response, with zero constraint violations, using arithmetic and six
measured numbers.

Any learned model has to beat that to justify existing. Neither of the 1-epoch
models does — on curtailment the fill scores 58.2 MW MAE against 150.3 and 159.2
for the two networks.
