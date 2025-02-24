import pygame
import random
import sys
import argparse
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
        str: One of 'UP', 'DOWN', 'LEFT', 'RIGHT'
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

    return moves.NONE  # Default if spider is already at fly position

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

            x, y = spider_pos

            if move == moves.UP:
                y -= 1
            elif move == moves.DOWN:
                y += 1
            elif move == moves.LEFT:
                x -= 1
            elif move == moves.RIGHT:
                x += 1

            spiders_pos[i] = (x, y)
            total_cost += 1

            if (x, y) in flies:
                flies.remove((x, y))

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
        print(f"Using previous move {prev_moves[i]} for spider {i}")
        if prev_moves[i] != moves.NONE:
            move_cost += 1
        spiders_pos[i] = env.apply_spider_move(spiders_pos[i], prev_moves[i])
        if spiders_pos[i] in flies:
            flies.remove(spiders_pos[i])
        i += 1

    print(f"Using spider move {spider_move} for spider {i}")
    if spider_move != moves.NONE:
        move_cost += 1
    spiders_pos[i] = env.apply_spider_move(spiders_pos[i], spider_move)
    if spiders_pos[i] in flies:
        flies.remove(spiders_pos[i])
    i += 1

    while i < len(spiders_pos):
        move = base_policy(spiders_pos[i], flies)
        print(f"Computed base policy move {move} for spider {i}")
        if move != moves.NONE:
            move_cost += 1

        spiders_pos[i] = env.apply_spider_move(spiders_pos[i], move)
        if spiders_pos[i] in flies:
            flies.remove(spiders_pos[i])
        i += 1

    return move_cost + alpha * base_policy_cost_to_go(spiders_pos, flies)

def base_player() -> int:
    """
    Run base policy for all spiders until all flies are eaten.
    Returns total cost for all spiders to eat all flies.

    Returns:
        int: Total cost for all spiders to eat all flies
    """
    running = True
    current_spider = 0

    running_cost = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return running_cost

        move = base_policy(env.spiders[current_spider], env.flies)
        if move != moves.NONE:
            running_cost += 1
        env.move_spider(current_spider, move)
        current_spider = (current_spider + 1) % len(env.spiders)

        # check if all flies are eaten
        if not any(cell for row in env.grid for cell in row):
            print("All flies are eaten!")
            running = False

        # wait for 1 second
        viz.update_display()
        pygame.time.wait(100)

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
            print("All flies are eaten!")
            running = False

        viz.update_display()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['manual', 'base', 'rollout', 'marollout'],
                       default='base', help='Mode: manual, base, rollout, or marollout')
    parser.add_argument('--seed', type=int, help='Random seed')
    args = parser.parse_args()

    if not args.seed:
        args.seed = random.randint(0, 1000000)

    print(f"Random seed: {args.seed}")
    random.seed(args.seed)

    # Initialize environment with 5 flies and 2 spiders
    env = GridEnvironment(k=5, spider_positions=[(6,0), (6,0)])
    viz = GridVisualization(env)

    pygame.event.set_allowed(None)
    pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])

    if args.mode == 'manual':
        interactive_player()
    elif args.mode == 'base':
        predicted_cost = base_policy_cost_to_go(env.spiders.copy(), env.flies.copy())
        actual_cost = base_player()

        print(f"Predicted cost: {predicted_cost}")
        print(f"Actual cost: {actual_cost}")
    elif args.mode == 'marollout':
        predicted_cost = move_cost_to_go(env.spiders.copy(), env.flies.copy(), spider_move=moves.UP)
        print(f"Predicted cost: {predicted_cost}")

    pygame.quit()
    sys.exit()
