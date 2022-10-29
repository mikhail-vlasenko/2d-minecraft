use std::fmt;
use std::hash::Hash;
use std::fmt::{Display, Formatter};
use crate::crafting::material::Material;

#[derive(Clone, Hash)]
pub struct Block {
    pub material: Material,
}


impl Display for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.material)
    }
}


impl PartialEq<Self> for Block {
    fn eq(&self, other: &Self) -> bool {
        self.material == other.material
    }
}

impl Eq for Block {}
