use std::fmt;
use std::fmt::Display;
use std::mem::take;
use serde::{Serialize, Deserialize};
use crate::map_generation::block::Block;
use crate::crafting::material::Material;
use crate::crafting::material::Material::*;
use crate::crafting::storable::Storable;


/// A square "column" on the field. Acts as a stack of Blocks.
/// Every field cell is made of it.
#[derive(PartialEq, Serialize, Deserialize, Debug, Clone)]
pub struct Tile {
    pub blocks: Vec<Block>,
    pub loot: Vec<Storable>,
}

impl Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.blocks.last().unwrap().material.glyph())
    }
}

impl Tile {
    pub fn new() -> Self {
        Tile {
            blocks: vec![],
            loot: vec![],
        }
    }
    pub fn with_blocks(blocks: Vec<Block>) -> Self {
        Tile {
            blocks,
            loot: vec![],
        }
    }
    pub fn from_material(material: Material) -> Self {
        Tile {
            blocks: vec![Block { material }],
            loot: vec![],
        }
    }
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
    pub fn top(&self) -> &Block {
        self.blocks.last().unwrap()
    }
    pub fn push(&mut self, block: Block) {
        self.blocks.push(block)
    }
    pub fn push_all(&mut self, blocks: Vec<Block>) {
        if self.blocks.len() + blocks.len() > 5 {
            panic!("Tried to push too many blocks to a tile")
        }
        self.blocks.extend(blocks)
    }
    pub fn pop(&mut self) -> Option<Block> {
        self.blocks.pop()
    }
    pub fn full(&self) -> bool {
        self.len() >= 5
    }
    pub fn top_material(&self) -> Material {
        self.top().material.clone()
    }
    pub fn get_loot(&self) -> &Vec<Storable> { &self.loot }
    pub fn add_loot(&mut self, new: Vec<Storable> ) {
        self.loot.extend(new)
    }
    pub fn gather_loot(&mut self) -> Vec<Storable> {
        take(&mut self.loot)
    }
    pub fn make_dirt() -> Tile {
        return Tile {
            blocks: vec![Block { material: Bedrock },
                         Block { material: Stone },
                         Block { material: Dirt }],
            loot: Vec::new(),
        };
    }
    pub fn add_tree(tile: &mut Tile) {
        if tile.top().material == Dirt && tile.len() < 4 {
            tile.blocks.push(Block { material: TreeLog });
            tile.blocks.push(Block { material: TreeLog });
        }
    }

    pub fn make_iron(tile: &mut Tile) {
        if tile.blocks[1].material == Stone {
            tile.blocks[1] = Block { material: IronOre };
            if tile.blocks[2].material == Stone {
                tile.blocks[2] = Block { material: IronOre };
            }
        }
    }

    pub fn make_diamond(tile: &mut Tile) {
        if tile.blocks[1].material == Stone {
            tile.blocks[1] = Block { material: DiamondOre };
        }
    }
    
    pub fn make_full_diamond(tile: &mut Tile) {
        if tile.blocks[1].material == DiamondOre && tile.blocks[2].material == Stone {
                tile.blocks[2] = Block { material: DiamondOre };
        }
    }

    pub fn make_rock(tile: &mut Tile) {
        if tile.top().material == Dirt {
            tile.pop();
            tile.push(Block { material: Stone });
        }
    }
}

/// Calls an augmentation function of the Tile with a certain probability
pub fn randomly_augment(tile: &mut Tile, augmentation: &dyn Fn(&mut Tile), proba: f32, rng: &mut impl rand::Rng) {
    let value: f32 = rng.gen();
    if value < proba {
        augmentation(tile)
    }
}
