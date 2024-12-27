use std::fmt;
use std::fmt::{Display, Formatter};
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};
use strum::IntoEnumIterator;
use Storable::*;
use crate::crafting::consumable::Consumable;
use crate::crafting::items::Item;
use crate::crafting::material::Material;
use crate::crafting::ranged_weapon::RangedWeapon;
use strum_macros::EnumIter;
use crate::crafting::interactable::InteractableKind;
use crate::crafting::texture_material::TextureMaterial;


/// Represents anything that can be stored in the inventory.
/// This includes all materials (even unbreakable), and all items.
#[derive(PartialEq, Copy, Clone, Hash, Serialize, Deserialize, Debug)]
pub enum Storable {
    M(Material),
    I(Item),
    C(Consumable),
    RW(RangedWeapon),
    IN(InteractableKind)
}

impl Storable {
    pub fn name(&self) -> &str {
        match self {
            M(mat) => mat.name(),
            I(item) => item.name(),
            C(cons) => cons.name(),
            RW(rw) => rw.name(),
            IN(inter) => inter.name(),
        }
    }

    pub fn craft_requirements(&self) -> &[(&Storable, u32)] {
        match self {
            M(mat) => mat.craft_requirements(),
            I(item) => item.craft_requirements(),
            C(cons) => cons.craft_requirements(),
            RW(rw) => rw.craft_requirements(),
            IN(inter) => inter.craft_requirements(),
        }
    }

    pub fn craft_yield(&self) -> u32 {
        match self {
            M(mat) => mat.craft_yield(),
            I(item) => item.craft_yield(),
            C(cons) => cons.craft_yield(),
            RW(rw) => rw.craft_yield(),
            IN(inter) => inter.craft_yield(),
        }
    }
    
    pub fn required_crafter(&self) -> Option<&Material> {
        match self {
            M(mat) => mat.required_crafter(),
            I(item) => item.required_crafter(),
            C(cons) => cons.required_crafter(),
            RW(rw) => rw.required_crafter(),
            IN(inter) => inter.required_crafter(),
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
        write!(f, "{}", self.name())
    }
}

#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Debug)]
pub enum CraftMenuSection {
    Placeables,
    Ingredients,
    Tools,
    Weapons,
    Interactables,
    Uncraftable,
}

impl CraftMenuSection {
    pub fn render_order() -> Vec<CraftMenuSection> {
        CraftMenuSection::iter().filter(|section| *section != CraftMenuSection::Uncraftable).collect()
    }
}

pub trait Craftable: Display + Into<Storable> + Copy {
    fn name(&self) -> &str;
    fn craft_requirements(&self) -> &[(&Storable, u32)];
    fn craft_yield(&self) -> u32;
    fn required_crafter(&self) -> Option<&Material> {
        match self {
            _ => None
        }
    }
    fn is_craftable(&self) -> bool {
        self.craft_yield() > 0
    }
    fn menu_section(&self) -> CraftMenuSection {
        match self {
            _ => CraftMenuSection::Uncraftable
        }
    }
}

impl Default for Storable {
    fn default() -> Self {
        M(Material::Dirt)
    }
}

lazy_static!{
    pub static ref ALL_STORABLES: Vec<Storable> = {
        let mut all = Vec::new();
        // only includes the 
        for mat in Material::iter() {
            match mat {
                Material::Texture(_) => continue,
                _ => all.push(M(mat))
            }
        }
        for texture in TextureMaterial::iter() {
            all.push(M(Material::Texture(texture)));
        }
        for item in Item::iter() {
            all.push(I(item));
        }
        for cons in Consumable::iter() {
            all.push(C(cons));
        }
        for rw in RangedWeapon::iter() {
            all.push(RW(rw));
        }
        for inter in InteractableKind::iter() {
            all.push(IN(inter));
        }
        all
    };
}

impl Storable {
    pub fn all() -> Vec<Storable> {
        ALL_STORABLES.clone()
    }
    pub fn craftables() -> Vec<Storable> {
        ALL_STORABLES.iter().filter(|s| s.is_craftable()).cloned().collect()
    }
}
