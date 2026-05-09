import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import networkx as nx
from collections import deque

plt.ion()


class GridWorld:
    def __init__(self, size=10, obstacle_density=0.25):
        self.size = size
        self.obstacle_density = obstacle_density

        self.grid = np.zeros((size, size), dtype=int)

        self.start = None
        self.goal = None

        self.generate_world()

    def generate_world(self):
        total_cells = self.size * self.size
        obstacle_count = int(total_cells * self.obstacle_density)

        obstacle_positions = random.sample(
            range(total_cells),
            obstacle_count
        )

        for pos in obstacle_positions:
            r = pos // self.size
            c = pos % self.size
            self.grid[r][c] = 1

        self.start = self.get_random_empty_cell()
        self.grid[self.start] = 2

        self.goal = self.get_random_empty_cell()
        self.grid[self.goal] = 3

    def get_random_empty_cell(self):
        while True:

            r = random.randint(0, self.size - 1)
            c = random.randint(0, self.size - 1)

            if self.grid[r][c] == 0:
                return (r, c)

    def draw(
        self,
        agent_pos=None,
        visited_nodes=None,
        depth_map=None,
        path_nodes=None
    ):

        # LEFT PANEL
        plt.subplot(1, 2, 1)
        plt.cla()

        display_grid = self.grid.copy()

        # BFS expansion coloring
        if visited_nodes and depth_map:

            for node in visited_nodes:

                if node == self.start or node == self.goal:
                    continue

                r, c = node

                depth = depth_map[node]

                # Extended depth range
                color_index = min(depth, 11)

                display_grid[r][c] = 4 + color_index

        cmap = plt.cm.colors.ListedColormap([
            "white",       # 0 empty
            "black",       # 1 obstacle
            "limegreen",   # 2 start
            "crimson",     # 3 goal

            # BFS wave colors
            "#ffe6f2",     # 4
            "#ffd1e8",     # 5
            "#f5ccff",     # 6
            "#e6ccff",     # 7
            "#d9d9ff",     # 8
            "#cce6ff",     # 9
            "#cceeff",     # 10
            "#ccfff2",     # 11
            "#e6ffcc",     # 12
            "#fff5cc",     # 13

            # Added depth colors
            "#ffe0cc",     # 14
            "#d9ffe6"      # 15
        ])

        plt.imshow(
            display_grid,
            cmap=cmap,
            vmin=0,
            vmax=15
        )

        # Final shortest path overlay
        if path_nodes:

            for node in path_nodes:

                if node == self.start or node == self.goal:
                    continue

                plt.scatter(
                    node[1],
                    node[0],
                    s=120,
                    marker='s'
                )

        # Agent marker
        if agent_pos:

            plt.scatter(
                agent_pos[1],
                agent_pos[0],
                s=200,
                marker='o'
            )

        plt.gca().invert_yaxis()

        plt.grid(True)
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))

        plt.title("BFS Grid Search")

        # Legend
        agent_legend = plt.Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markersize=8,
            label='Agent'
        )

        path_legend = plt.Line2D(
            [0], [0],
            marker='s',
            linestyle='None',
            markersize=8,
            label='Final Path'
        )

        legend_items = [

            agent_legend,

            mpatches.Patch(
                color='black',
                label='Obstacle'
            ),

            mpatches.Patch(
                color='limegreen',
                label='Start'
            ),

            mpatches.Patch(
                color='crimson',
                label='Goal'
            ),

            mpatches.Patch(
                color='#ffe6f2',
                label='BFS Early'
            ),

            mpatches.Patch(
                color='#e6ccff',
                label='BFS Mid'
            ),

            mpatches.Patch(
                color='#cceeff',
                label='BFS Outer'
            ),

            mpatches.Patch(
                color='#ffe0cc',
                label='BFS Deep'
            ),

            mpatches.Patch(
                color='#d9ffe6',
                label='BFS Deepest'
            ),

            path_legend
        ]

        plt.legend(
            handles=legend_items,
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8
        )

    def in_bounds(self, position):
        row, col = position

        return (
            0 <= row < self.size and
            0 <= col < self.size
        )

    def is_walkable(self, position):
        row, col = position

        return self.grid[row][col] != 1

    def get_neighbors(self, position):
        row, col = position

        possible_moves = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]

        valid_neighbors = []

        for move in possible_moves:

            if self.in_bounds(move) and self.is_walkable(move):
                valid_neighbors.append(move)

        return valid_neighbors


class Agent:
    def __init__(self, start_position):

        self.position = start_position

        # Persistent BFS data
        self.visited_nodes = set()
        self.depth_map = {}

        # Search tree
        self.search_tree = nx.DiGraph()
        self.search_tree.add_node(start_position)

    def bfs_search(self, world):

        queue = deque()
        parent = {}

        visited = self.visited_nodes
        depth_map = self.depth_map

        queue.append(self.position)

        visited.add(self.position)

        depth_map[self.position] = 0

        while queue:

            current = queue.popleft()

            self.position = current

            world.draw(
                agent_pos=self.position,
                visited_nodes=visited,
                depth_map=depth_map
            )

            self.draw_tree()

            plt.draw()
            plt.pause(0.05)

            # Goal found
            if current == world.goal:

                print("\nGoal found!")

                return self.reconstruct_path(
                    parent,
                    world.goal
                )

            neighbors = world.get_neighbors(current)

            for neighbor in neighbors:

                if neighbor not in visited:

                    visited.add(neighbor)

                    parent[neighbor] = current

                    depth_map[neighbor] = (
                        depth_map[current] + 1
                    )

                    self.search_tree.add_edge(
                        current,
                        neighbor
                    )

                    queue.append(neighbor)

        print("\nNo path found.")

        return None

    def reconstruct_path(self, parent_map, goal):

        path = []

        current = goal

        while current in parent_map:

            path.append(current)

            current = parent_map[current]

        path.append(current)

        path.reverse()

        return path

    def draw_tree(self):

        # RIGHT PANEL
        plt.subplot(1, 2, 2)
        plt.cla()

        pos = {}

        levels = {}

        root = list(self.search_tree.nodes())[0]

        levels[root] = 0

        for node in self.search_tree.nodes():

            if node == root:
                continue

            parents = list(
                self.search_tree.predecessors(node)
            )

            if parents:

                levels[node] = (
                    levels[parents[0]] + 1
                )

        layer_counts = {}

        for node, level in levels.items():

            if level not in layer_counts:
                layer_counts[level] = 0

            x = layer_counts[level]
            y = -level

            pos[node] = (x, y)

            layer_counts[level] += 1

        nx.draw(
            self.search_tree,
            pos,
            with_labels=True,
            node_size=350,
            font_size=5,
            arrows=False
        )

        plt.title("BFS Search Tree")


def main():

    plt.figure(figsize=(16, 6))

    world = GridWorld(size=10)

    print("Start:", world.start)
    print("Goal:", world.goal)

    agent = Agent(world.start)

    final_path = agent.bfs_search(world)

    if final_path:

        print("\nShortest Path:")
        print(final_path)

        # Final state keeps BFS history
        world.draw(
            agent_pos=world.goal,
            visited_nodes=agent.visited_nodes,
            depth_map=agent.depth_map,
            path_nodes=final_path
        )

        agent.draw_tree()

        plt.draw()

        print(
            f"\nPath Length: {len(final_path)}"
        )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()