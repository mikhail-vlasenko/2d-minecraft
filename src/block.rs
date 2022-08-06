use std::fmt;
use std::hash::Hash;
use std::fmt::{Display, Formatter};
use crate::material::Material;

#[derive(Clone, Hash)]
pub struct Block {
    pub material: &'static Material<'static>, // todo: what if 'static here?
}


impl Display for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.material)
    }
}


impl PartialEq<Self> for Block {
    fn eq(&self, other: &Self) -> bool {
        self.material.name == other.material.name
    }
}

impl Eq for Block {}
