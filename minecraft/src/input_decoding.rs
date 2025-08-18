use winit::keyboard::KeyCode;
use game_logic::auxiliary::actions::Action;
use game_logic::character::player::Player;

pub fn map_key_to_action(key: &KeyCode, player: &Player) -> Option<Action> {
    match key {
        KeyCode::KeyW => Some(Action::WalkNorth),
        KeyCode::KeyA => Some(Action::WalkWest),
        KeyCode::KeyS => Some(Action::WalkSouth),
        KeyCode::KeyD => Some(Action::WalkEast),
        KeyCode::ArrowLeft => Some(Action::TurnLeft),
        KeyCode::ArrowRight => Some(Action::TurnRight),
        KeyCode::KeyQ => Some(Action::Mine),
        KeyCode::KeyE => Some(Action::Place),
        KeyCode::KeyC => Some(Action::Craft),
        KeyCode::KeyF => Some(Action::Consume),
        KeyCode::KeyX => Some(Action::Shoot),
        KeyCode::KeyM => Some(Action::ToggleMap),
        KeyCode::Space => Some(Action::ToggleCraftMenu),
        KeyCode::Escape => {
            // if player is interacting, close the interaction, otherwise toggle the main menu
            if player.get_interacting_with().is_some() {
                Some(Action::CloseInteractableMenu)
            } else {
                Some(Action::ToggleMainMenu)
            }
        },
        _ => None,
    }
}
