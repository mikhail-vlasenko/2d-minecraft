use strum::IntoEnumIterator;
use crate::auxiliary::actions::Action;
use crate::crafting::consumable::Consumable;
use crate::crafting::interactable::InteractableKind;
use crate::crafting::material::Material;
use crate::crafting::texture_material::TextureMaterial;


/// These conversions are used primarily for the FFI.

impl From<i32> for Action {
    fn from(i: i32) -> Self {
        use crate::auxiliary::actions::Action::*;
        match i {
            0 => WalkNorth,
            1 => WalkWest,
            2 => WalkSouth,
            3 => WalkEast,
            4 => TurnLeft,
            5 => TurnRight,
            6 => Mine,
            7 => Place,
            8 => Craft,
            9 => Consume,
            10 => Shoot,
            11 => CloseInteractableMenu,
            12 => ToggleMap,
            13 => ToggleCraftMenu,
            14 => ToggleMainMenu,
            _ => {
                let i = i as usize - 15;
                let mut materials = Material::iter();
                if i < materials.len() {
                    let material = materials
                        .nth(i)
                        .expect("Material index out of bounds");
                    return PlaceSpecificMaterial(material)
                }
                
                let i = i - materials.len();
                let mut interactables = InteractableKind::iter();
                if i < interactables.len() {
                    let interactable = interactables
                        .nth(i)
                        .expect("Interactable index out of bounds");
                    return PlaceSpecificInteractable(interactable)
                }
                
                let i = i - interactables.len();
                
                // todo Craftables for crafting
                
                let mut consumables = Consumable::iter();
                if i < consumables.len() {
                    let consumable = consumables
                        .nth(i)
                        .expect("Consumable index out of bounds");
                    return ConsumeSpecific(consumable)
                }
                unreachable!()
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
            Diamond => 7,
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
