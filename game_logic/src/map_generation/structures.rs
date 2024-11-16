use rand::Rng;
use crate::map_generation::tile::{Tile};
use crate::crafting::material::Material;
use crate::crafting::texture_material::TextureMaterial;
use crate::SETTINGS;
use crate::map_generation::chunk::Chunk;

use Material::*;
use TextureMaterial::*;
use crate::crafting::items::Item;

impl Chunk {
    pub fn add_structures(&mut self, chunk_rng: &mut impl Rng) {
        let structure = Self::choose_structure(chunk_rng);
        if structure.is_none() {
            return;
        }
        let structure = structure.unwrap();

        for _ in 0..10 {
            // todo: rotation and flip
            let structure_width = structure[0].len();
            let structure_height = structure.len();
            let structure_x = chunk_rng.gen_range(0..self.get_size() - structure_height);
            let structure_y = chunk_rng.gen_range(0..self.get_size() - structure_width);
            let mut structure_valid = true;
            for i in 0..structure_width {
                for j in 0..structure_height {
                    if self.len_at((structure_x + i) as i32, (structure_y + j) as i32) != 3 {
                        structure_valid = false;
                    }
                }
            }
            if structure_valid {
                for i in 0..structure_width {
                    for j in 0..structure_height {
                        self.push_all_at(structure[i][j].blocks.clone(), (structure_x + i) as i32, (structure_y + j) as i32);
                        self.add_loot_at(structure[i][j].loot.clone(), (structure_x + i) as i32, (structure_y + j) as i32);
                    }
                }
                break;
            }
        }
    }

    fn choose_structure(chunk_rng: &mut impl Rng) -> Option<Vec<Vec<Tile>>> {
        let value: f32 = chunk_rng.gen();
        if value < SETTINGS.read().unwrap().field.generation.structures.robot_proba {
            Some(Self::make_war_robot())
        } else {
            None
        }
    }

    fn make_war_robot() -> Vec<Vec<Tile>> {
        let mut structure = vec![];
        structure.push(vec![Tile::from_material(Texture(RobotTL)), Tile::from_material(Texture(RobotTR))]);
        structure.push(vec![Tile::from_material(Texture(RobotBL)), Tile::from_material(Texture(RobotBR))]);
        structure[0][1].add_loot(vec![Item::TargetingModule.into()]);
        structure
    }
}


