# prey-predator-marl

Predators learning to hunt, in a 16×16 grid with walls and a bolt-hole.

This is a rewrite of my undergraduate thesis code. Two predators learn by
tabular Q-learning; three prey follow a fixed script — sit still until they spot
a threat, then run for the lair if one's in sight, otherwise just run.

I rewrote it because the original had a bug I didn't find until much later: the
predators moved twice per step, once from the learned policy and once from a
hardcoded chase routine I'd left in. So the results measured a hand-written
chaser about as much as a learned one.

## Running it

```bash
pip install -e ".[dev]"
python -c "import prey_predator.train"
```

Everything lands in `runs/run_NNN/` — the Q-tables, the metrics, and the config
that produced them.

## How it works

Predators and prey both move one cell in any of eight directions. A predator
catches a prey by landing on it; a prey escapes by reaching a lair. Either way
that prey is out, and the episode ends when none are left.

The interesting part is what a predator can see. Not the whole grid — just where
the nearest visible prey and lair are *relative to itself*. "Prey two east, one
north" is the same situation whether you're in the corner or the middle, so one
lesson covers the whole map. It also keeps the Q-table at about 15,000 entries
instead of millions.

## Something I got wrong

The first reward scheme paid predators for being close to prey. Reasonable
enough, except the best way to collect that reward turns out to be standing next
to a prey forever and never catching it — a catch removes the prey, and with it
the income. The agents worked this out. Capture rate dropped while reward
climbed, which looks like success if you only plot reward.

The fix is to pay for *closing distance* rather than *being close*. Hovering
then earns nothing. Same gradient, no exploit.

## References

- Lenzitti, Tegolo & Valenti (2005), *Prey-Predator Strategies in a Multiagent System* — the paper this environment comes from
- Kok & Vlassis (2004), *Sparse Cooperative Q-learning*