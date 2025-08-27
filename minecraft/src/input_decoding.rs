use winit::keyboard::KeyCode;
use game_logic::auxiliary::actions::Action;
use game_logic::character::abilities::Ability::{Barricade, Charge, SecondWind, WhirlAttack};

pub fn map_key_to_action(key: &KeyCode, is_player_interacting: bool) -> Option<Action> {
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
        KeyCode::Digit1 => Some(Action::UseAbility(WhirlAttack)),
        KeyCode::Digit2 => Some(Action::UseAbility(Charge)),
        KeyCode::Digit3 => Some(Action::UseAbility(Barricade)),
        KeyCode::Digit4 => Some(Action::UseAbility(SecondWind)),
        KeyCode::KeyM => Some(Action::ToggleMap),
        KeyCode::Space => Some(Action::ToggleCraftMenu),
        KeyCode::Escape => {
            // if player is interacting, close the interaction, otherwise toggle the main menu
            if is_player_interacting {
                Some(Action::CloseInteractableMenu)
            } else {
                Some(Action::ToggleMainMenu)
            }
        },
        _ => None,
    }
}
