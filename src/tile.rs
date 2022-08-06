use std::fmt;
use std::fmt::Display;
use crate::block::Block;

#[derive(Clone)]
pub struct Tile {
    pub blocks: Vec<Block>,
    pub top: usize
}

impl Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.blocks[self.top].material.display_symbol)
    }
}
