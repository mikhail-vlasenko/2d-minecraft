use crate::Field;
use crate::map_generation::mobs::priority_queue::PriorityQueue;

pub struct AStar {
    pending: PriorityQueue<(i32, i32)>,
    visited: Vec<Vec<bool>>,
    dist_to: Vec<Vec<i32>>,
    parent: Vec<Vec<(i32, i32)>>,
    min_loaded: (i32, i32),
    max_loaded: (i32, i32),
    directions: Vec<(i32, i32)>,
}

impl AStar {
    pub fn new(field: &Field) -> Self {
        let size = field.loaded_tiles_size();
        let min_loaded = field.min_loaded_idx();
        let max_loaded = field.max_loaded_idx();
        let directions = vec![(0, 1), (1, 0), (0, -1), (-1, 0)];

        let pending = PriorityQueue::new();
        let visited = vec![vec![false; size]; size];
        let dist_to = vec![vec![0; size]; size];
        let parent = vec![vec![(0, 0); size]; size];
        Self {
            pending,
            visited,
            dist_to,
            parent,
            min_loaded,
            max_loaded,
            directions,
        }
    }

    /// Produces the next step (direction) according to a*.
    pub fn a_star(&mut self, field: &Field, source: (i32, i32), destination: (i32, i32)) -> (i32, i32) {
        let max_priority = Self::max_acceptable_priority(source, destination);
        self.visit(source);
        self.pending.push(0, source);

        while self.pending.len() > 0 {
            let mut current_tile = self.pending.pop().unwrap();

            if current_tile == destination {
                // restore path
                if current_tile == source {
                    println!("destination reached");
                    return (0, 0);
                }

                loop {
                    let curr_idx = self.convert_idx(current_tile);
                    let prev = self.parent[curr_idx.0][curr_idx.1];
                    if prev == source {
                        return (current_tile.0 - source.0, current_tile.1 - source.1);
                    }
                    current_tile = prev;
                }
            }
            let idx = self.convert_idx(current_tile);

            let neighbours = self.get_neighbours(field, current_tile);
            for n in neighbours {
                self.set_parent(n, current_tile);
                let priority = self.dist_to[idx.0][idx.1] + estimate_remaining(n, destination);
                if priority <= max_priority {
                    self.visit(current_tile);
                    self.pending.push(priority, n);
                }
            }
        }
        println!("no route found");
        (0, 0)
    }

    fn convert_idx(&self, tile: (i32, i32)) -> (usize, usize) {
        ((tile.0 - self.min_loaded.0) as usize, (tile.1 - self.min_loaded.1) as usize)
    }

    fn visit(&mut self, tile: (i32, i32)) {
        let idx = self.convert_idx(tile);
        self.visited[idx.0][idx.1] = true;
    }

    fn set_parent(&mut self, tile: (i32, i32), parent_tile: (i32, i32)) {
        let idx = self.convert_idx(tile);
        let parent_idx = self.convert_idx(parent_tile);
        self.parent[idx.0][idx.1] = parent_tile;
        self.dist_to[idx.0][idx.1] = self.dist_to[parent_idx.0][parent_idx.1] + 1;
    }

    fn get_neighbours(&self, field: &Field, tile: (i32, i32)) -> Vec<(i32, i32)> {
        let this_height = field.len_at(tile.0, tile.1);
        let mut res = Vec::new();

        for d in &self.directions {
            let new_pos = (tile.0 + d.0, tile.1 + d.1);
            // in bounds, not too high, and not visited
            if self.min_loaded.0 <= new_pos.0 && new_pos.0 <= self.max_loaded.0 &&
                self.min_loaded.1 <= new_pos.1 && new_pos.1 <= self.max_loaded.1 &&
                can_step(field, tile, new_pos, this_height) {
                let idx = self.convert_idx(new_pos);
                if !self.visited[idx.0][idx.1] {
                    res.push(new_pos);
                }
            }
        }
        res
    }

    /// Priority higher than this results in a route that is too long
    fn max_acceptable_priority(source: (i32, i32), destination: (i32, i32)) -> i32 {
        estimate_remaining(source, destination) + 10
    }

}

fn can_step(field: &Field, source: (i32, i32), destination: (i32, i32), current_height: usize) -> bool {
    field.len_at(destination.0, destination.1) <= current_height + 1 &&
        !field.mob_at(destination.0, destination.1)
}

fn estimate_remaining(tile: (i32, i32), destination: (i32, i32)) -> i32 {
    (tile.0 - destination.0).abs() + (tile.1 - destination.1).abs()
}

