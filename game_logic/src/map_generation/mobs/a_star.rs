use std::path::PathBuf;
use rand::random;
use serde::{Serialize, Deserialize};
use crate::character::player::Player;
use crate::map_generation::field::Field;
use crate::map_generation::field::DIRECTIONS;
use crate::map_generation::mobs::priority_queue::PriorityQueue;
use crate::map_generation::save_load::{load_game, save_game};
use crate::SETTINGS;

/// Runs A* on the field. Cuts off execution if expected route is too long.
#[derive(Serialize, Deserialize, Debug)]
pub struct AStar {
    size: usize,
    radius: i32,
    pending: PriorityQueue<(i32, i32)>,
    visited: Vec<Vec<bool>>,
    dist_to: Vec<Vec<i32>>,
    parent: Vec<Vec<(i32, i32)>>,
    min_loaded: (i32, i32),
    max_loaded: (i32, i32),
}

impl AStar {
    pub fn new(radius: i32) -> Self {
        let size = (radius as usize) * 2 + 1;
        let min_loaded = (0, 0);
        let max_loaded = (0, 0);

        let pending = PriorityQueue::new();
        let visited = vec![vec![false; size]; size];
        let dist_to = vec![vec![0; size]; size];
        let parent = vec![vec![(0, 0); size]; size];
        Self {
            size,
            radius,
            pending,
            visited,
            dist_to,
            parent,
            min_loaded,
            max_loaded,
        }
    }

    /// Produces the next step (direction) according to a*, and the route length.
    pub fn full_pathing(&mut self, field: &Field, 
                        source: (i32, i32), destination: (i32, i32), 
                        max_detour: i32) -> ((i32, i32), i32) {
        let max_priority = Self::shortest_len(source, destination) + max_detour;

        let idx = self.convert_idx(source);
        self.dist_to[idx.0][idx.1] = 0;
        self.pending.push(0, source);

        while self.pending.len() > 0 {
            let mut current_tile = self.pending.pop().unwrap();
            self.visit(current_tile);

            if current_tile == destination {
                // restore path
                if current_tile == source {
                    return ((0, 0), 0);
                }

                let mut route_len = 1;
                loop {
                    let curr_idx = self.convert_idx(current_tile);
                    let prev = self.parent[curr_idx.0][curr_idx.1];
                    if prev == source {
                        return ((current_tile.0 - source.0, current_tile.1 - source.1), route_len);
                    }
                    current_tile = prev;
                    route_len += 1;
                    if route_len > max_priority {
                        println!("\nThis should not happen. Here is the info for debugging convenience:\n");
                        println!("Max detour: {}", max_detour);
                        println!("Max priority: {}", max_priority);
                        println!("Source: {:?}", source);
                        println!("Destination: {:?}", destination);
                        let mut dummy_player = Player::new(field);
                        dummy_player.x = destination.0;
                        dummy_player.y = destination.1;
                        dummy_player.land(field);
                        let mut path = PathBuf::from(SETTINGS.read().unwrap().save_folder.clone().into_owned());
                        let num: u16 = random();
                        path.push(format!("a_star_error_{}", num));
                        save_game(field, &dummy_player, &path);
                        println!("Saved the game to {:?}", path);
                        unreachable!("A* backtracking loop should finish before reaching this point");
                    }
                }
            }
            let idx = self.convert_idx(current_tile);

            let neighbours = self.get_neighbours(field, current_tile);
            for n in neighbours {
                let priority = self.dist_to[idx.0][idx.1] + estimate_remaining(n, destination);
                if priority <= max_priority {
                    self.set_parent(n, current_tile);
                    self.pending.push(priority, n);
                }
            }
        }
        ((0, 0), 0)
    }

    /// A* works on a potentially smaller part of the field, then all loaded chunks.
    /// This part should be symmetric around the player.
    pub fn reset(&mut self, player_x: i32, player_y: i32) {
        self.visited = vec![vec![false; self.size]; self.size];
        self.pending = PriorityQueue::new();
        self.min_loaded = (player_x - self.radius, player_y - self.radius);
        self.max_loaded = (player_x + self.radius, player_y + self.radius);
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
        let mut res = Vec::new();

        for d in &DIRECTIONS {
            let new_pos = (tile.0 + d.0, tile.1 + d.1);
            // in bounds, not too high, has no mob, and not visited
            if self.min_loaded.0 <= new_pos.0 && new_pos.0 <= self.max_loaded.0 &&
                self.min_loaded.1 <= new_pos.1 && new_pos.1 <= self.max_loaded.1 &&
                can_step(field, tile, new_pos) {
                let idx = self.convert_idx(new_pos);
                if !self.visited[idx.0][idx.1] {
                    res.push(new_pos);
                }
            }
        }
        res
    }

    fn shortest_len(source: (i32, i32), destination: (i32, i32)) -> i32 {
        estimate_remaining(source, destination)
    }

    pub fn get_radius(&self) -> i32 {
        self.radius
    }
}

impl AStar {
    pub fn default() -> Self {
        Self {
            size: 0,
            radius: 0,
            pending: PriorityQueue::new(),
            visited: Vec::new(),
            dist_to: Vec::new(),
            parent: Vec::new(),
            min_loaded: (0, 0),
            max_loaded: (0, 0),
        }
    }
}

pub fn can_step(field: &Field, source: (i32, i32), destination: (i32, i32)) -> bool {
    field.len_at(destination) <= field.len_at(source) + 1 &&
        !field.is_occupied(destination)
}

pub fn can_step_towards(field: &Field, source: (i32, i32), destination: (i32, i32)) -> bool {
    let diff = (destination.0 - source.0, destination.1 - source.1);
    let can1 = diff.0 != 0 && can_step(field, source, (source.0 + diff.0, source.1));
    let can2 = diff.1 != 0 && can_step(field, source, (source.0, source.1 + diff.1));
    can1 || can2
}

fn estimate_remaining(tile: (i32, i32), destination: (i32, i32)) -> i32 {
    (tile.0 - destination.0).abs() + (tile.1 - destination.1).abs()
}

#[test]
#[ignore]
fn test_hanging_from_save() {
    // let mut path = PathBuf::from(SETTINGS.read().unwrap().save_folder.clone().into_owned());
    // path.push("a_star_error_15079");
    // let (mut field, mut player) = load_game(path.as_path());
    // field.a_star = AStar::new(SETTINGS.read().unwrap().pathing.a_star_radius);
    // println!("{:?}", player.x);
    // println!("{:?}", player.y);
    // let source = (-25, -12);
    // let destination = (-20, -14);
    // let (dir, len) = field.full_pathing(source, destination, (player.x, player.y), None);
    // assert!(len > 0);
    // assert_ne!(dir, (0, 0));
}

