use std::fmt;
use std::hash::Hash;
use std::fmt::{Display, Formatter};
use crate::material::Material;

#[derive(Clone, Hash)]
pub struct Block<'a> {
    pub material: &'a Material<'a>,
}


impl Display for Block<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.material)
    }
}


impl PartialEq<Self> for Block<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.material.name == other.material.name
    }
}

impl Eq for Block<'_> {}
