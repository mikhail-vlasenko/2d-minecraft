use strum::IntoEnumIterator;
use crate::auxiliary::actions::Action;
use crate::crafting::consumable::Consumable;
use crate::crafting::interactable::InteractableKind;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::crafting::texture_material::TextureMaterial;


/// These conversions are used primarily for the FFI.

impl TryFrom<i32> for Action {
    type Error = &'static str;
    fn try_from(i: i32) -> Result<Self, Self::Error> {
        use crate::auxiliary::actions::Action::*;
        match i {
            0 => Ok(WalkNorth),
            1 => Ok(WalkWest),
            2 => Ok(WalkSouth),
            3 => Ok(WalkEast),
            4 => Ok(TurnLeft),
            5 => Ok(TurnRight),
            6 => Ok(Mine),
            7 => Ok(Place),
            8 => Ok(Craft),
            9 => Ok(Consume),
            10 => Ok(Shoot),
            11 => Ok(CloseInteractableMenu),
            12 => Ok(ToggleMap),
            13 => Ok(ToggleCraftMenu),
            14 => Ok(ToggleMainMenu),
            _ => {
                let i = i as usize - 15;
                let mut materials = Material::iter();
                if i < materials.len() {
                    let material = materials
                        .nth(i).unwrap();
                    return Ok(PlaceSpecificMaterial(material))
                }
                
                let i = i - materials.len();
                let mut interactables = InteractableKind::iter();
                if i < interactables.len() {
                    let interactable = interactables
                        .nth(i).unwrap();
                    return Ok(PlaceSpecificInteractable(interactable))
                }
                
                let i = i - interactables.len();
                let craftables = Storable::craftables();
                if i < craftables.len() {
                    let craftable = craftables[i];
                    return Ok(CraftSpecific(craftable))
                }
                
                let i = i - craftables.len();
                let mut consumables = Consumable::iter();
                if i < consumables.len() {
                    let consumable = consumables
                        .nth(i).unwrap();
                    return Ok(ConsumeSpecific(consumable))
                }
                Err("unknown action")
            },
        }
    }
}

impl Into<i32> for Material {
    fn into(self) -> i32 {
        use crate::crafting::material::Material::*;
        match self {
            Dirt => 0,
            TreeLog => 1,
            Plank => 2,
            Stone => 3,
            Bedrock => 4,
            IronOre => 5,
            CraftTable => 6,
            DiamondOre => 7,
            Texture(t) => t as i32 + 8,
        }
    }
}

impl Into<i32> for TextureMaterial {
    fn into(self) -> i32 {
        use crate::crafting::texture_material::TextureMaterial::*;
        match self {
            Unknown => 0,
            RobotTL => 1,
            RobotTR => 2,
            RobotBL => 3,
            RobotBR => 4,
        }
    }
}
