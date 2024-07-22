use std::cell::RefCell;
use crate::auxiliary::actions::Action;
use crate::auxiliary::actions::Action::*;
use crate::character::player::Player;
use crate::crafting::storable::Storable;
use crate::map_generation::field::Field;

/// Makes an action, corresponding to the key.
/// Returns how much turn was used.
pub fn act(action: &Action, player: &mut Player, field: &mut Field, craft_menu_open: &RefCell<bool>, main_menu_open: &RefCell<bool>) -> f32 {
    match action {
        WalkNorth | WalkWest | WalkSouth | WalkEast => player.walk(action, field),
        TurnLeft => player.turn(1),
        TurnRight => player.turn(-1),
        Mine => player.mine_infront(field),
        Place => player.place_current(field),
        Craft => player.craft_current(field),
        Consume => player.consume_current(),
        Shoot => player.shoot_current(field),
        CloseInteractableMenu => { player.interacting_with = None; 0. },
        ToggleMap => { player.toggle_map(); 0. },
        ToggleCraftMenu => { craft_menu_open.replace(!craft_menu_open.take()); 0. },
        ToggleMainMenu => { main_menu_open.replace(!main_menu_open.take()); 0. },
        PlaceSpecificMaterial(material) => {
            player.placement_storable = Storable::M(material.clone());
            player.place_current(field)
        }
        PlaceSpecificInteractable(interactable) => {
            player.placement_storable = Storable::IN(interactable.clone());
            player.place_current(field)
        }
        CraftSpecific(storable) => {
            player.crafting_item = storable.clone();
            player.craft_current(field)
        }
        ConsumeSpecific(consumable) => {
            player.consumable = consumable.clone();
            player.consume_current()
        }
    }
}
