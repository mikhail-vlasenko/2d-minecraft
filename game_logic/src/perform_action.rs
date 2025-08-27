use std::cell::RefCell;
use crate::auxiliary::actions::Action;
use crate::auxiliary::actions::Action::*;
use crate::character::p2p_interactions::InteractionStore;
use crate::character::player::Player;
use crate::crafting::storable::Storable;
use crate::map_generation::field::Field;
use crate::map_generation::mobs::mob::Position;

/// Makes an action, corresponding to the key.
/// Returns how much turn was used.
pub fn act(
    action: &Action, player: &mut Player, field: &mut Field, other_positions: &Vec<Position>, 
    craft_menu_open: &RefCell<bool>, main_menu_open: &RefCell<bool>
) -> (f32, InteractionStore) {
    match action {
        WalkNorth | WalkWest | WalkSouth | WalkEast => player.walk(action, field, other_positions),
        TurnLeft => player.turn(1),
        TurnRight => player.turn(-1),
        Mine => player.mine_infront(field),
        Place => player.place_current(field),
        Craft => player.craft_current(field),
        Consume => player.consume_current(),
        Shoot => player.shoot_current(field, other_positions),
        CloseInteractableMenu => { player.stop_interacting(); (0., vec![]) },
        ToggleMap => { player.toggle_map(); (0., vec![]) },
        ToggleCraftMenu => { craft_menu_open.replace(!craft_menu_open.take()); (0., vec![]) },
        ToggleMainMenu => { main_menu_open.replace(!main_menu_open.take()); (0., vec![]) },
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
        UseAbility(ability) => player.use_ability(ability, field, other_positions),
    }
}
