# New-Repo Plan: Torch-Policy RAP Bridge

## Objective
Reimplement the RAP bridge in a new repository with minimal upstream coupling by:

1. running policy inference in Python/Torch from a `.pt` checkpoint
2. stepping simulation through upstream `Drive.step(actions)`
3. converting simulator outputs to RAP scenario input and rendering frames

This avoids carrying a custom native-policy runtime in C for the bridge workflow.

## Reference Pattern from `pufferl`
This design is directly inspired by the existing `pufferl` evaluate loop pattern:

1. `policy.forward_eval(observations, state)`
2. sample action from logits
3. `env.step(action)`

I will keep that same control structure, but attach RAP conversion/rendering in each frame iteration.

## What I Will Change (New Repo Only)
- Add a Torch-policy bridge runner that:
  - loads policy checkpoint (`.pt`)
  - infers actions each frame in Python
  - steps PufferDrive via `Drive.step`
  - reads global state/map and builds RAP scenario
  - renders/saves frame outputs
- Keep bridge logic in Python only.

## What I Will Not Change (Upstream Core)
- No modifications to upstream C simulator/binding internals:
  - `pufferlib/ocean/drive/drive.h`
  - `pufferlib/ocean/env_binding.h`
  - `pufferlib/ocean/drive/binding.c`
  - `pufferlib/ocean/drive/drive.c`
  - `pufferlib/ocean/drive/visualize.c`

## Runtime Flow in the New Bridge
Per-frame flow:

`Torch policy inference` -> `sample action` -> `Drive.step(action)` -> `get global state/map` -> `RAP scenario conversion` -> `render/save`

Important clarification:
- only the policy inference location changes (native C policy path to Python/Torch)
- simulator transition logic is still upstream C through `Drive.step` (same stepping semantics)

## Why This Is Minimal
- zero new C API surface
- maximum reuse of upstream stable env interface
- clearer maintenance boundary (bridge orchestration in Python, simulator in upstream C)


## Goals
This plan is complete when:

1. Torch-policy bridge runs end-to-end for target frame count
2. simulator stepping goes through upstream `Drive.step` only
3. no upstream C files need to be modified
4. RAP outputs are produced from simulator state in the new repo
