use std::cell::RefCell;
use crate::auxiliary::actions::Action;
use crate::character::player::Player;
use crate::crafting::storable::Storable;
use crate::map_generation::field::Field;

/// Makes an action, corresponding to the key.
/// Returns how much turn was used.
pub fn act(action: &Action, player: &mut Player, field: &mut Field, craft_menu_open: &RefCell<bool>, main_menu_open: &RefCell<bool>) -> f32 {
    match action {
        Action::WalkNorth => player.walk("w", field),
        Action::WalkWest => player.walk("a", field),
        Action::WalkSouth => player.walk("s", field),
        Action::WalkEast => player.walk("d", field),
        Action::TurnLeft => player.turn(1),
        Action::TurnRight => player.turn(-1),
        Action::Mine => player.mine_infront(field),
        Action::Place => player.place_current(field),
        Action::Craft => player.craft_current(field),
        Action::Consume => player.consume_current(),
        Action::Shoot => player.shoot_current(field),
        Action::CloseInteractableMenu => { player.interacting_with = None; 0. },
        Action::ToggleMap => { player.toggle_map(); 0. },
        Action::ToggleCraftMenu => { craft_menu_open.replace(!craft_menu_open.take()); 0. },
        Action::ToggleMainMenu => { main_menu_open.replace(!main_menu_open.take()); 0. },
        Action::PlaceSpecificMaterial(material) => {
            player.placement_storable = Storable::M(material.clone());
            player.place_current(field)
        }
        Action::PlaceSpecificInteractable(interactable) => {
            player.placement_storable = Storable::IN(interactable.clone());
            player.place_current(field)
        }
        Action::CraftSpecific(storable) => {
            player.crafting_item = storable.clone();
            player.craft_current(field)
        }
        Action::ConsumeSpecific(consumable) => {
            player.consumable = consumable.clone();
            player.consume_current()
        }
    }
}
