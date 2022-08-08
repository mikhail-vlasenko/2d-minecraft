use std::fmt;
use std::fmt::Display;
use crate::block::Block;
use crate::materials::*;

#[derive(Clone)]
pub struct Tile {
    pub blocks: Vec<Block>,
}

impl Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.blocks.last().unwrap().material.display_symbol)
    }
}

impl Tile {
    pub fn top(&self) -> &Block {
        self.blocks.last().unwrap()
    }
    pub fn push(&mut self, block: Block) {
        self.blocks.push(block)
    }
    pub fn pop(&mut self) -> Option<Block> {
        self.blocks.pop()
    }
    pub fn make_dirt() -> Tile {
        return Tile {
            blocks: vec![Block { material: &BEDROCK },
                         Block { material: &STONE },
                         Block { material: &DIRT }],
        };
    }
    pub fn make_stone() -> Tile {
        return Tile {
            blocks: vec![Block { material: &BEDROCK },
                         Block { material: &STONE },
                         Block { material: &STONE }],
        };
    }
    pub fn add_tree(tile: &mut Tile) {
        if tile.top().material == &DIRT {
            tile.blocks.push(Block { material: &TREE_LOG });
            tile.blocks.push(Block { material: &TREE_LOG });
        }
    }
}

pub fn randomly_add(tile: &mut Tile, addition: &dyn Fn(&mut Tile), proba: f32) {
    let rng: f32 = rand::random();
    if rng < proba {
        addition(tile)
    }
}
