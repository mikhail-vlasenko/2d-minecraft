use std::fmt;
use std::fmt::Display;
use crate::block::Block;

#[derive(Clone)]
pub struct Tile<'a> {
    pub blocks: Vec<Block<'a>>,
    pub top: usize
}

impl Display for Tile<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.blocks[self.top].material.display_symbol)
    }
}
