# Spiders and Flies Simulation

A grid-based environment where spiders chase flies using different movement policies. This project demonstrates various planning algorithms in a simple game-like setting.

<img src="https://github.com/mikemykhaylov/spiders-n-flies/blob/3d3f737c19d07b59247e585473ac8e1a4140a117/public/screenshot.png" width="50%" alt="Spiders and Flies">

## Overview

This simulation places spiders and flies on a grid. The goal is for the spiders to catch all flies with minimal movement. The project implements and compares different movement policies:

- **Base Policy**: Simple greedy approach where each spider moves toward its nearest fly
- **Regular Rollout**: Planning joint actions for all spiders at once
- **Multi-agent Rollout (MARollout)**: One-step rollout for each spider considering previous spiders' moves

## Requirements

- Python 3.10+
- PyGame
- tqdm (for testing)

## Installation

```bash
# Clone the repository
git clone https://github.com/mikemykhaylov/spiders-n-flies.git
cd spiders-n-flie

# Install dependencies
pip install pygame tqdm
```

## Usage

Run the simulation with different policies:

```bash
# Base policy
python spiders/index.py --mode base --show

# Multiple Action Rollout policy
python spiders/index.py --mode marollout --show

# Joint Rollout policy
python spiders/index.py --mode rollout --show

# Manual mode (control with arrow keys)
python spiders/index.py --mode manual

# Run comparative test (no visualization)
python spiders/index.py --mode test
```

### Command Line Options

- `-m, --mode`: Choose policy mode (`manual`, `base`, `rollout`, `marollout`, `test`)
- `-s, --show`: Show visualization (not available in test mode)
- `--seed`: Set random seed for reproducibility

## Environment

The simulation consists of:
- A 10x10 grid
- Multiple spiders (default: 2)
- Multiple flies (default: 5)
- Spiders can move up, down, left, or right within grid boundaries
- When a spider lands on a fly, the fly is consumed

## Policy Descriptions

### Base Policy
- Each spider moves toward the nearest fly using Manhattan distance
- Prefers horizontal movement when distances are equal
- Simple but not optimal for multiple spiders

### Joint Rollout
- Plans actions for all spiders simultaneously
- Evaluates all possible move combinations
- Most computationally expensive but closest to optimal

### Multi-agent Rollout (MARollout)
- Plans one move at a time for each spider
- Takes into account moves made by previous spiders in the current turn
- Uses the base policy for future spider moves
- Better than base policy but still myopic

## Performance Comparison

The `test` mode runs thousands of simulations and reports:
- Average cost (moves) for each policy
- Average computation time for each policy

Typically, the policies perform in this order (from worst to best):
1. Base Policy (fastest)
2. MARollout (balanced)
3. Joint Rollout (most optimal but slowest)

## License

[MIT License](LICENSE)
