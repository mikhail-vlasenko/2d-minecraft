use crate::auxiliary::actions::Action;
use crate::crafting::material::Material;
use crate::crafting::texture_material::TextureMaterial;


/// These conversions are used primarily for the FFI.

impl Into<i32> for Action {
    fn into(self) -> i32 {
        use crate::auxiliary::actions::Action::*;
        match self {
            WalkNorth => 0,
            WalkWest => 1,
            WalkSouth => 2,
            WalkEast => 3,
            TurnLeft => 4,
            TurnRight => 5,
            Mine => 6,
            Place => 7,
            Craft => 8,
            Consume => 9,
            Shoot => 10,
            CloseInteractableMenu => 11,
            ToggleMap => 12,
            ToggleCraftMenu => 13,
            ToggleMainMenu => 14,
        }
    }
}

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
            _ => unreachable!(),
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
