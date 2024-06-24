use strum_macros::EnumIter;
use serde::{Serialize, Deserialize};


#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum StatusEffect{
    Speedy,
}
