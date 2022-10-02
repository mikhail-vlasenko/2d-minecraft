use std::fmt;
use std::fmt::Display;
use crate::block::Block;
use crate::Material::*;


/// A square "column" on the field. Acts as a stack of Blocks.
/// Every field cell is made of it.
#[derive(Clone)]
pub struct Tile {
    pub blocks: Vec<Block>,
}

impl Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.blocks.last().unwrap().material.glyph())
    }
}

impl Tile {
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
    pub fn top(&self) -> &Block {
        self.blocks.last().unwrap()
    }
    pub fn push(&mut self, block: Block) {
        self.blocks.push(block)
    }
    pub fn pop(&mut self) -> Option<Block> {
        self.blocks.pop()
    }
    pub fn full(&self) -> bool {
        self.len() >= 5
    }
    pub fn make_dirt() -> Tile {
        return Tile {
            blocks: vec![Block { material: Bedrock },
                         Block { material: Stone },
                         Block { material: Dirt }],
        };
    }
    pub fn make_stone() -> Tile {
        return Tile {
            blocks: vec![Block { material: Bedrock },
                         Block { material: Stone },
                         Block { material: Stone }],
        };
    }
    pub fn add_tree(tile: &mut Tile) {
        if tile.top().material == Dirt {
            tile.blocks.push(Block { material: TreeLog });
            tile.blocks.push(Block { material: TreeLog });
        }
    }
}

/// Calls an augmentation function of the Tile with a certain probability
pub fn randomly_augment(tile: &mut Tile, augmentation: &dyn Fn(&mut Tile), proba: f32) {
    let rng: f32 = rand::random();
    if rng < proba {
        augmentation(tile)
    }
}
