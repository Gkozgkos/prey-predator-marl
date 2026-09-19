# prey-predator-marl

Two predators, three prey, one bolt-hole, and a 16×16 grid full of walls.
The predators learn to hunt with plain tabular Q-learning. The prey don't learn
anything: they sit still until they spot a predator, then run for the lair if
they can see it, and just run otherwise.

This started as my undergraduate thesis. Rereading the code a while later, I
found out it didn't do what I thought it did. After every learned move, a bit
of leftover code moved each predator a second time, straight at the nearest
prey if it could see one, or in a random direction if it couldn't. So my
predators were secretly twice as fast, could move diagonally, and had a free
search strategy, and my results said more about that script than about
anything the agents learned. This repo is the rewrite.

## Running it

```bash
pip install -e ".[dev]"
python -m prey_predator.train
```

Each run gets its own folder under `runs/` with the Q-tables, the training
metrics and the settings used.

To look at a trained run, open `src/prey_predator/evaluate.py`, point `RUN` at
the folder, and run the file. With `WATCH = True` you get a window and can
watch an episode play out. With `WATCH = False` it plays 200 episodes on the
same fixed set of maps and prints the catch rate at different amounts of
randomness.

## How it works

Everyone moves one cell per step, in any of eight directions. About a third of
the grid is wall, reshuffled every episode, and walls block sight as well as
movement, so a predator can creep up on a prey from behind cover. Landing on a
prey catches it; a prey that reaches the lair is safe. Either way it's out of
the game, and the episode ends when all of them are gone.

A predator doesn't see the grid. It only knows where the nearest prey and the
lair are relative to itself, if they're in view. That keeps things small, about
15,000 possible situations, and it means a lesson learned in one corner of the
map works everywhere else.

## The reward trap

My first attempt paid predators a little for being close to a prey. They
figured out quickly that the best way to get paid was to stand next to a prey
and never catch it, because catching it ends the payments. Catches went down
while reward went up, which looks great on a reward plot and is completely
useless.

Paying for getting *closer* instead of *being close* fixed it. Standing still
earns nothing, so there's nothing to farm.

## The corner problem

Watching the agents turned up the next problem. When a predator can't see
anything, every empty patch of grid looks the same to it, so it has exactly one
move for "I see nothing", and it makes that move forever. In practice that
means walking into a corner and staying there.

Mixing a small amount of randomness into its choices shows how much this
costs. Same trained agents, same 200 maps, before walls blocked sight:

| Randomness | Catches per episode |
|---|---|
| none | 0.19 |
| 5% | 0.70 |
| 100% (just wandering) | 0.72 |

Once it can get unstuck, the learned policy catches about as many prey as
random wandering does. More training won't help, since the agents have already
seen almost every situation they can tell apart. They need to see more: whether
there's a wall next to them, and which way they were heading. That's what I'm
working on next.

These numbers come from a single training run, so take them as a rough
picture, not a final result.

## Reading

- Lenzitti, Tegolo & Valenti (2005), *Prey-Predator Strategies in a Multiagent System*. The paper this environment is based on.
- Kok & Vlassis (2004), *Sparse Cooperative Q-learning*.
- Singh, Jaakkola & Jordan (1994), *Learning Without State-Estimation in Partially Observable Markovian Decision Processes*. Why an agent that can't tell situations apart gets stuck.

## License

MIT