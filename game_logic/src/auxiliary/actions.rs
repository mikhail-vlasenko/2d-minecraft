use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;
use crate::crafting::consumable::Consumable;
use crate::crafting::interactable::InteractableKind;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;

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
}
