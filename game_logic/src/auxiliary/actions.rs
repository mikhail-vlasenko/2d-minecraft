use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;
use crate::character::abilities::Ability;
use crate::character::player::Player;
use crate::crafting::consumable::Consumable;
use crate::crafting::interactable::InteractableKind;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;
use crate::crafting::storable::Storable::{C, I, IN, M, RW};
use crate::map_generation::field::Field;

#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum Action {
    WalkNorth,
    WalkWest,
    WalkSouth,
    WalkEast,
    TurnLeft,
    TurnRight,
    Mine,
    Place,
    Craft,
    Consume,
    Shoot,
    CloseInteractableMenu,
    ToggleMap,
    ToggleCraftMenu,
    ToggleMainMenu,
    // these actions can be used when selection of the active item is complicated
    PlaceSpecificMaterial(Material),
    PlaceSpecificInteractable(InteractableKind),
    CraftSpecific(Storable),
    ConsumeSpecific(Consumable),
    
    UseAbility(Ability),
}

impl Action {
    pub fn ffi_disabled(&self) -> bool {
        use Action::*;
        match self {
            CloseInteractableMenu | ToggleMap | ToggleCraftMenu | ToggleMainMenu => true,
            _ => false,
        }
    
    }
}

/// Returns whether the player can take the action.
/// Built for FFI, so actions like place and craft are unavailable in favour of PlaceSpecificMaterial and CraftSpecific.
/// Menu action are thus also disabled.
pub fn can_take_action(action: &Action, player: &Player, field: &Field) -> bool {
    use Action::*;
    match action {
        WalkNorth | WalkWest | WalkSouth | WalkEast => {
            player.can_walk(action, field)
        } 
        TurnLeft | TurnRight => true, 
        Mine => {
            let mat = field.top_material_at(player.coords_infront());
            mat.required_mining_power() <= player.get_mining_power()
        }  
        Place | Craft | Consume=> {
            false
        }
        Shoot => {
            player.has(&RW(player.ranged_weapon)) && player.has(&I(*player.ranged_weapon.ammo()))
        } 
        CloseInteractableMenu | ToggleMap | ToggleCraftMenu | ToggleMainMenu=> {
            false
        } 
        PlaceSpecificMaterial(material) => {
            player.has(&M(*material))
        } 
        PlaceSpecificInteractable(interactable) => {
            player.has(&IN(*interactable))
        }
        CraftSpecific(storable) => {
            player.can_craft(storable, field).is_ok()
        } 
        ConsumeSpecific(consumable) => {
            player.has(&C(*consumable))
        },
        UseAbility(ability) => {
            player.is_ability_ready(ability)
        }
    }
}
