use std::fmt;
use std::fmt::{Display, Formatter};
use crate::items::Item;
use crate::material::Material;
use Storable::*;


#[derive(PartialEq, Copy, Clone, Hash)]
pub enum Storable {
    M(Material),
    I(Item)
}

impl Storable {
    pub fn name(&self) -> &str {
        match self {
            M(mat) => mat.name(),
            I(item) => item.name()
        }
    }

    pub fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            M(mat) => mat.craft_requirements(),
            I(item) => item.craft_requirements()
        }
    }

    pub fn craft_yield(&self) -> u32 {
        match self {
            M(mat) => mat.craft_yield(),
            I(item) => item.craft_yield()
        }
    }

    pub fn is_craftable(&self) -> bool {
        self.craft_yield() > 0
    }
}

impl TryFrom<String> for Storable {
    type Error = &'static str;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let res = Material::try_from(value.clone());
        match res {
            Ok(mat) => return Ok(M(mat)),
            _ => 0
        };
        let res = Item::try_from(value.clone());
        match res {
            Ok(item) => return Ok(I(item)),
            _ => 0
        };
        return Err("unknown material")
    }
}

impl Display for Storable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            M(mat) => mat.name(),
            I(item) => item.name()
        })
    }
}