use crate::material::Material;

#[derive(Clone)]
pub struct Block<'a> {
    pub material: &'a Material<'a>,
}