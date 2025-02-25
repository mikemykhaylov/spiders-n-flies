import pygame
import random
import sys
import argparse
import itertools

from game import GridEnvironment, GridVisualization, moves
from typing import List, Tuple

def base_policy(spider_pos: Tuple[int, int], flies: set[Tuple[int, int]]) -> moves:
    """
    Determine spider's move towards nearest fly using Manhattan distance.
    Prefers horizontal movement over vertical when distances are equal.

    Args:
        spider_pos: Tuple of (x, y) coordinates for spider position
        flies: Set of (x, y) coordinates for all flies

    Returns:
        str: One of 'UP', 'DOWN', 'LEFT', 'RIGHT', or 'NONE' moves
    """
    spider_x, spider_y = spider_pos
    min_distance = float('inf')
    nearest_fly = None

    # Find nearest fly using Manhattan distance
    for fly_pos in sorted(flies):
        fly_x, fly_y = fly_pos
        distance = abs(spider_x - fly_x) + abs(spider_y - fly_y)
        if distance < min_distance:
            min_distance = distance
            nearest_fly = fly_pos
        elif distance == min_distance and nearest_fly:
            # Prefer horizontal movement over vertical when distances are equal
            if abs(nearest_fly[0] - spider_x) < abs(fly_x - spider_x):
                nearest_fly = fly_pos

    if nearest_fly is None:
        return moves.NONE

    fly_x, fly_y = nearest_fly

    # Determine direction to move
    dx = fly_x - spider_x
    dy = fly_y - spider_y

    # Prefer horizontal movement over vertical when possible
    if dx != 0:
        return moves.RIGHT if dx > 0 else moves.LEFT
    if dy != 0:
        return moves.DOWN if dy > 0 else moves.UP

    # should never reach here
    print(f"Spider at {spider_pos} is already at fly position {nearest_fly}")
    raise ValueError("Spider is already at fly position")

def base_policy_cost_to_go(spiders_pos: List[Tuple[int, int]], flies: set[Tuple[int, int]]) -> int:
    """
    Calculate total cost for all spiders to eat all flies using base policy.

    Args:
        spiders_pos: List of spider positions as (x, y) coordinates
        flies: Set of (x, y) coordinates for all flies

    Returns:
        int: Total cost for all spiders to eat all flies
    """
    total_cost = 0

    while flies:
        for i in range(len(spiders_pos)):
            spider_pos = spiders_pos[i]

            move = base_policy(spider_pos, flies)
            if move == moves.NONE:
                # No flies left to eat
                break

            spiders_pos[i] = GridEnvironment.apply_spider_move(spider_pos, move)
            total_cost += 1

            if spiders_pos[i] in flies:
                flies.remove(spiders_pos[i])


    return total_cost

def move_cost_to_go(spiders_pos: List[Tuple[int, int]], flies: set[Tuple[int, int]], prev_moves: List[moves] = [], spider_move: moves = moves.NONE, alpha: float = 1.0) -> float:
    """
    Calculates g(state, (...prev_moves, spider_move, ..base_policy_moves)) for base policy.
    First, it moves the spider according to the spider_move, then runs the base policy for all remaining spiders.
    Finally, it calculates the cost for all spiders to eat all flies.

    Args:
        spiders_pos: List of spider positions as (x, y) coordinates
        flies: Set of (x, y) coordinates for all flies
        prev_moves: List of previous moves for spiders during this action
        spider_move: Move for the current spider
        alpha: Discount factor for future costs

    Returns:
        int: Total cost for all spiders to eat all flies
    """
    move_cost = 0

    i = 0
    while i < len(prev_moves):
        # print(f"Using previous move {prev_moves[i]} for spider {i}")
        if prev_moves[i] != moves.NONE:
            move_cost += 1
        spiders_pos[i] = GridEnvironment.apply_spider_move(spiders_pos[i], prev_moves[i])
        if spiders_pos[i] in flies:
            flies.remove(spiders_pos[i])
        i += 1

    # print(f"Using spider move {spider_move} for spider {i}")
    if spider_move != moves.NONE:
        move_cost += 1
    spiders_pos[i] = GridEnvironment.apply_spider_move(spiders_pos[i], spider_move)
    if spiders_pos[i] in flies:
        flies.remove(spiders_pos[i])
    i += 1

    while i < len(spiders_pos):
        move = base_policy(spiders_pos[i], flies)
        # print(f"Computed base policy move {move} for spider {i}")
        if move != moves.NONE:
            move_cost += 1

        spiders_pos[i] = GridEnvironment.apply_spider_move(spiders_pos[i], move)
        if spiders_pos[i] in flies:
            flies.remove(spiders_pos[i])
        i += 1

    # print(f"g(x, {spider_move}) = {move_cost}, J(f({spiders_pos}, {flies})) = {base_policy_cost_to_go(spiders_pos.copy(), flies.copy())}")

    return move_cost + base_policy_cost_to_go(spiders_pos.copy(), flies.copy())

def marollout_policy(spiders_pos: List[Tuple[int, int]], flies: set[Tuple[int, int]], prev_moves: List[moves] = []) -> moves:
    """
    Find best move by minimizing over all possible moves for the current spider with a rollout policy.

    Args:
        spiders_pos: List of spider positions as (x, y) coordinates
        flies: Set of (x, y) coordinates for all flies
        prev_moves: List of previous moves for spiders during this action

    Returns:
        str: One of 'UP', 'DOWN', 'LEFT', 'RIGHT', or 'NONE' moves
    """
    min_cost = float('inf')
    best_move = moves.NONE

    for move in moves:
        cost = move_cost_to_go(spiders_pos.copy(), flies.copy(), prev_moves.copy(), move)
        if cost < min_cost:
            min_cost = cost
            best_move = move

    return best_move

def rollout_policy(spiders_pos: List[Tuple[int, int]], flies: set[Tuple[int, int]]) -> List[moves]:
    """
    Finds best combination of moves for all spiders by minimizing over all possible moves for all spiders.

    Args:
        spiders_pos: List of spider positions as (x, y) coordinates
        flies: Set of (x, y) coordinates for all flies

    Returns:
        List[moves]: List of moves for all spiders
    """
    n = len(spiders_pos)
    min_cost = float('inf')
    best_moves = [moves.NONE] * len(spiders_pos)

    for move_combination in itertools.product(moves, repeat=len(spiders_pos)):
        cost = move_cost_to_go(spiders_pos.copy(), flies.copy(), [move for move in move_combination[:n - 1]], move_combination[n-1])
        if cost < min_cost:
            min_cost = cost
            best_moves = [move for move in move_combination]

    return best_moves

def base_player(env: GridEnvironment, show: bool) -> int:
    """
    Run base policy for all spiders until all flies are eaten.
    Returns total cost for all spiders to eat all flies.

    Arguments:
        env: GridEnvironment -- The environment to run the policy on
        show: bool -- Whether to show the visualization

    Returns:
        int: Total cost for all spiders to eat all flies
    """
    running = True

    running_cost = 0

    while running:
        if show:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return running_cost

        for current_spider in range(len(env.spiders)):
            move = base_policy(env.spiders[current_spider], env.flies)
            if move != moves.NONE:
                running_cost += 1
            env.move_spider(current_spider, move)

            # check if all flies are eaten
            if not any(cell for row in env.grid for cell in row):
                running = False
                break

        if show:
            viz.update_display()
            env.wait()

    return running_cost

def marollout_player(env: GridEnvironment, show: bool) -> int:
    """
    Run base policy for all spiders until all flies are eaten.
    Returns total cost for all spiders to eat all flies.

    Arguments:
        env: GridEnvironment -- The environment to run the policy on
        show: bool -- Whether to show the visualization

    Returns:
        int: Total cost for all spiders to eat all flies
    """
    running = True

    running_cost = 0
    prev_moves = []

    while running:
        if show:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return running_cost

        spiders, flies = env.spiders.copy(), env.flies.copy()
        for current_spider in range(len(env.spiders)):
            move = marollout_policy(spiders, flies, prev_moves)
            if move != moves.NONE:
                running_cost += 1
            env.move_spider(current_spider, move)

            # check if all flies are eaten
            if not any(cell for row in env.grid for cell in row):
                running = False
                break

            if len(prev_moves) == len(env.spiders) - 1:
                prev_moves = []
            else:
                prev_moves.append(move)

        if show:
            viz.update_display()
            env.wait()

    return running_cost

def rollout_player(env: GridEnvironment, show: bool) -> int:
    """
    Run base policy for all spiders until all flies are eaten.
    Returns total cost for all spiders to eat all flies.

    Arguments:
        env: GridEnvironment -- The environment to run the policy on
        show: bool -- Whether to show the visualization

    Returns:
        int: Total cost for all spiders to eat all flies
    """
    running = True
    running_cost = 0

    while running:
        if show:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return running_cost

        spiders, flies = env.spiders.copy(), env.flies.copy()
        spider_moves = rollout_policy(spiders, flies)

        for i in range(len(env.spiders)):
            move = spider_moves[i]
            if move != moves.NONE:
                running_cost += 1
            env.move_spider(i, move)

            # check if all flies are eaten
            if not any(cell for row in env.grid for cell in row):
                # print("All flies are eaten!")
                running = False
                break

        if show:
            viz.update_display()
            env.wait()

    return running_cost

def interactive_player():
    running = True
    current_spider = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    env.move_spider(current_spider, moves.UP)
                    current_spider = (current_spider + 1) % len(env.spiders)
                elif event.key == pygame.K_DOWN:
                    env.move_spider(current_spider, moves.DOWN)
                    current_spider = (current_spider + 1) % len(env.spiders)
                elif event.key == pygame.K_LEFT:
                    env.move_spider(current_spider, moves.LEFT)
                    current_spider = (current_spider + 1) % len(env.spiders)
                elif event.key == pygame.K_RIGHT:
                    env.move_spider(current_spider, moves.RIGHT)
                    current_spider = (current_spider + 1) % len(env.spiders)

        # check if all flies are eaten
        if not any(cell for row in env.grid for cell in row):
            # print("All flies are eaten!")
            running = False

        viz.update_display()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['manual', 'base', 'rollout', 'marollout', 'test'],
                       default='base', help='Mode: manual, base, rollout, or marollout')
    parser.add_argument('-s', '--show', action='store_true', help='Show visualization')
    parser.add_argument('-t', '--test', action='store_true', help='Run tests')
    parser.add_argument('--seed', type=int, help='Random seed')
    args = parser.parse_args()
    if args.mode == 'manual':
        # set show to True for manual mode
        args.show = True

    if not args.seed:
        args.seed = random.randint(0, 1000000)
    if args.mode != 'test':
        print(f"Random seed: {args.seed}")
    random.seed(args.seed)

    # Initialize environment with 5 flies and 2 spiders
    env = GridEnvironment(k=5, spider_positions=[(6,0), (6,0)], wait_delay=100)

    if args.show:
        viz = GridVisualization(env)
        pygame.event.set_allowed(None)
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])
        viz.update_display()

    match args.mode:
        case 'manual':
            interactive_player()
        case 'base':
            predicted_cost = base_policy_cost_to_go(env.spiders.copy(), env.flies.copy())
            actual_cost = base_player(env, args.show)

            print(f"Predicted cost: {predicted_cost}")
            print(f"Actual cost: {actual_cost}")
        case 'marollout':
            predicted_cost = base_policy_cost_to_go(env.spiders.copy(), env.flies.copy())
            actual_cost = marollout_player(env, args.show)

            print(f"Predicted cost: {predicted_cost}")
            print(f"Actual cost: {actual_cost}")
        case 'rollout':
            predicted_cost = base_policy_cost_to_go(env.spiders.copy(), env.flies.copy())
            actual_cost = rollout_player(env, args.show)

            print(f"Predicted cost: {predicted_cost}")
            print(f"Actual cost: {actual_cost}")
        case 'test':
            for i in range(100000):
                seed = random.randint(0, 100000000)
                random.seed(seed)
                env.reset(k=5, spider_positions=[(6,0), (6,0)])
                if i % 10000 == 0:
                    print(env.flies)
                predicted_cost = base_policy_cost_to_go(env.spiders.copy(), env.flies.copy())
                actual_cost = base_player()
                if predicted_cost != actual_cost:
                    print(f"Seed: {seed}, Predicted cost: {predicted_cost}, Actual cost: {actual_cost}")
                    break

    pygame.quit()
    sys.exit()
