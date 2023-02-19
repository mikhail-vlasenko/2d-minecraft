use strum_macros::EnumIter;


#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Debug)]
pub enum StatusEffect{
    Speedy,
}
