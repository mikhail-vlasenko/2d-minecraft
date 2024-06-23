use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

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
}
