use std::cmp::min;
use std::f32::consts::PI;
use serde::{Serialize, Deserialize};
use crate::auxiliary::actions::Action;
use crate::auxiliary::animations::{AnimationsBuffer, ProjectileType};
use crate::character::status_effects::StatusEffect;
use crate::crafting::consumable::Consumable;
use crate::crafting::interactable::{Interactable, InteractableKind};
use crate::map_generation::block::Block;
use crate::map_generation::field::{AbsoluteChunkPos, AbsolutePos, Field};
use crate::crafting::inventory::Inventory;
use crate::crafting::items::Item::Arrow;
use crate::crafting::material::Material;
use crate::crafting::material::Material::*;
use crate::crafting::ranged_weapon::RangedWeapon;
use crate::crafting::storable::Storable;
use crate::crafting::storable::Storable::{I, RW};
use crate::SETTINGS;


/// The player.
#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct Player {
    pub x: i32,
    pub y: i32,
    pub z: usize,
    rotation: f32,
    hp: i32,
    inventory: Inventory,
    /// list of status effects and their remaining duration
    status_effects: Vec<(StatusEffect, i32)>,

    // Storables that will be used for the corresponding actions
    // Are set though UI
    pub placement_storable: Storable,
    pub crafting_item: Storable,
    pub consumable: Consumable,
    pub ranged_weapon: RangedWeapon,
    /// The absolute map location of the interactable the player is currently interacting with.
    pub interacting_with: Option<(i32, i32)>,
    pub viewing_map: bool,

    pub message: String,
    pub score: i32,
    pub animations_buffer: AnimationsBuffer,
}

impl Player {
    pub fn new(field: &Field) -> Self {
        let mut player = Self {
            x: 0,
            y: 0,
            z: 0,
            rotation: 0.,
            hp: SETTINGS.read().unwrap().player.max_hp,
            inventory: Inventory::new(),
            status_effects: Vec::new(),
            placement_storable: Plank.into(),
            crafting_item: Storable::M(Plank),
            consumable: Consumable::Apple,
            ranged_weapon: RangedWeapon::Bow,
            interacting_with: None,
            viewing_map: false,
            message: String::new(),
            score: 0,
            animations_buffer: AnimationsBuffer::new(),
        };
        player.land(field);
        player
    }
    /// Breaks the top block or picks up the interactable at the given position.
    /// Returns how much action was spent.
    pub fn mine(&mut self, field: &mut Field, pos: AbsolutePos) -> f32 {
        if self.interacting_with.is_some() {
            self.interacting_with = None;
        }

        if field.get_interactable_kind_at(pos).is_some() {
            let kind = field.break_interactable_at(pos);
            self.pickup(kind.into(), 1);
            self.add_message(&format!("Picked up {}", kind));
            return 0.;
        }
        let mat =  field.top_material_at(pos);
        if mat.required_mining_power() > self.get_mining_power() {
            self.add_message(&format!("Need {} mining PWR", mat.required_mining_power()));
            0.
        } else {
            self.add_message(&format!("Mined {}", mat));
            self.score_mined(&mat);
            if mat.drop_item().is_some() {
                self.pickup(mat.drop_item().unwrap(), 1);
            } else {
                self.pickup(mat.into(), 1);
            }
            field.pop_at(pos);
            self.get_speed_multiplier()
        }
    }

    /// Mines a block in front of the player.
    pub fn mine_infront(&mut self, field: &mut Field) -> f32 {
        self.mine(field, self.coords_infront())
    }

    pub fn get_mining_power(&self) -> i32 {
        self.inventory.mining_power()
    }

    /// places a block of that material at position, deduces from inventory
    fn place_material(&mut self, field: &mut Field, pos: (i32, i32), material: Material) -> f32 {
        if field.full_at(pos) {
            self.add_message(&format!("Too high to build"));
            return 0.;
        }

        let placement_block = Block { material };

        if self.inventory.drop(&material.into(), 1) {
            field.push_at(placement_block, pos);
            self.get_speed_multiplier()
        } else {
            self.add_message(&format!("You do not have a block of {}", material));
            0.
        }
    }

    /// Places a new Interactable of InteractableKind at position, deduces from inventory
    fn place_interactable(&mut self, field: &mut Field, pos: (i32, i32), interactable: InteractableKind) -> f32 {
        if field.get_interactable_kind_at(pos).is_some() {
            self.add_message(&format!("Already has an interactable"));
            return 0.;
        }

        if self.inventory.drop(&interactable.into(), 1) {
            field.add_interactable(Interactable::new(interactable, pos));
            self.get_speed_multiplier()
        } else {
            self.add_message(&format!("You do not have {}", interactable));
            0.
        }
    }

    fn interact(&mut self, field: &mut Field, pos: (i32, i32)) {
        if self.interacting_with.is_some() {
            self.interacting_with = None;
            return;
        }
        self.interacting_with = Some(pos);
    }

    /// Places a currently selected block in front of the player.
    /// If there is an interactable in front of the player, interact with it instead.
    /// Returns how much action was spent.
    pub fn place_current(&mut self, field: &mut Field) -> f32 {
        let placement_pos = self.coords_infront();
        if field.get_interactable_kind_at(placement_pos).is_some() {
            self.interact(field, placement_pos);
            0.
        } else if field.is_occupied(placement_pos) {
            self.add_message(&format!("Can't place on a mob"));
            0.
        } else {
            match self.placement_storable {
                Storable::M(material) => self.place_material(field, placement_pos, material),
                Storable::IN(interactable) => {
                    self.place_interactable(field, placement_pos, interactable)
                },
                _ => unreachable!("Can't place {}", self.placement_storable),
            }
        }
    }

    pub fn walk_delta(&self, walk_action: &Action) -> (i32, i32) {
        match walk_action {
            Action::WalkNorth => (-1, 0),
            Action::WalkWest => (0, -1),
            Action::WalkSouth => (1, 0),
            Action::WalkEast => (0, 1),
            _ => unreachable!()
        }
    }
    
    pub fn walk(&mut self, walk_action: &Action, field: &mut Field) -> f32 {
        let delta = self.walk_delta(walk_action);
        self.step(field, delta)
    }

    /// Moves the Player by (delta_x, delta_y), checking for height conditions.
    ///
    /// # Arguments
    ///
    /// * `field`: the playing field
    /// * `delta`:
    ///
    /// returns: how much action was spent.
    fn step(&mut self, field: &mut Field, delta: (i32, i32)) -> f32 {
        if self.interacting_with.is_some() {
            self.interacting_with = None;
        }
        let new_pos = (self.x + delta.0, self.y + delta.1);
        if field.len_at(new_pos) <= self.z + 1 {
            // fighting
            if field.is_occupied(new_pos) {
                self.damage_mob(field, new_pos, self.get_melee_damage());
                return self.get_speed_multiplier()
            }
            // movement
            self.x += delta.0;
            self.y += delta.1;
            self.land(field);

            // loot
            let loot = field.gather_loot_at((self.x, self.y));
            if loot.len() > 0 {
                for l in loot {
                    self.add_message(&format!("Looted {}", l));
                    self.pickup(l, 1);
                }
            }

            // chunk loading
            let curr_chunk: AbsoluteChunkPos = (field.chunk_pos(self.x), field.chunk_pos(self.y));
            if curr_chunk != field.get_central_chunk() {
                field.load(curr_chunk.0, curr_chunk.1);
            }
            self.get_speed_multiplier()
        } else {
            self.add_message(&"Too high to step on");
            0.
        }
    }

    /// Sets the z coordinate of the Player
    pub fn land(&mut self, field: &Field) {
        self.z = field.len_at((self.x, self.y));
    }

    fn exists_around(&self, field: &Field, material: &Material) -> bool {
        let x_indices = vec![-1, 0, 1];
        let y_indices = vec![-1, 0, 1];
        for x in x_indices {
            for y in &y_indices {
                if &field.top_material_at((self.x + x, self.y + y)) == material {
                    return true;
                }
            }
        }
        return false;
    }

    /// Returns Ok if the player can craft the item, 
    /// else returns an error with message about why the item can not be crafted.
    pub fn can_craft(&self, item: &Storable, field: &Field) -> Result<bool, String> {
        for (req, amount) in item.craft_requirements() {
            if self.inventory.count(&req) < *amount {
                return Err(format!("Need {} of {}, have {}", amount, req, self.inventory.count(&req)));
            }
        }

        let crafter = item.required_crafter();
        let crafter_near =
            if crafter == None { true } else {
                self.exists_around(field, crafter.unwrap())
            };
        if !crafter_near {
            return Err(format!("There is no {} nearby", crafter.unwrap()));
        }

        Ok(true)
    }

    /// Returns true if the player has all the ingredients to craft the item,
    /// but not necessarily can craft it now.
    pub fn has_all_ingredients(&self, item: &Storable) -> bool {
        if !item.is_craftable() {
            return false;
        }
        for (req, amount) in item.craft_requirements() {
            if self.inventory.count(&req) < *amount {
                return false;
            }
        }
        true
    }

    /// If crafting the given item is possible, subtracts the ingredients and adds the item to the inventory.
    /// Else does nothing.
    pub fn craft(&mut self, item: Storable, field: &Field) -> f32 {
        if let Err(message) = self.can_craft(&item, field) {
            self.add_message(&message);
            return 0.
        }
        for (req, amount) in item.craft_requirements() {
            self.inventory.drop(req, *amount);
        }
        let craft_yield = item.craft_yield();
        self.inventory.pickup(item, craft_yield);
        self.add_message(&format!("Crafted {} of {}", craft_yield, item));
        self.score_crafted(&item, craft_yield);
        self.get_speed_multiplier()
    }

    pub fn craft_current(&mut self, field: &Field) -> f32 {
        self.craft(self.crafting_item, field)
    }

    pub fn pickup(&mut self, storable: Storable, amount: u32) {
        self.inventory.pickup(storable, amount)
    }

    pub fn consume(&mut self, consumable: Consumable) -> f32 {
        if self.inventory.drop(&Storable::C(consumable), 1) {
            consumable.apply_effect(self);
            self.get_speed_multiplier()
        } else {
            self.add_message(&format!("You dont have {}", consumable));
            0.
        }
    }

    pub fn consume_current(&mut self) -> f32 {
        self.consume(self.consumable)
    }

    pub fn shoot(&mut self, field: &mut Field, weapon: RangedWeapon) -> f32 {
        if !self.has(&RW(weapon)) {
            self.add_message(&format!("You do not have {}", weapon));
            return 0.;
        }
        if !self.inventory.drop(&I(*weapon.ammo()), 1) {
            self.add_message(&format!("No ammo! ({})", weapon.ammo()));
            return 0.;
        }
        let direction = self.coords_from_rotation();
        let mut curr_tile = (self.x, self. y);
        let height = self.z + 1;
        for _ in 0..weapon.range() {
            curr_tile = (curr_tile.0 + direction.0, curr_tile.1 + direction.1);
            if field.len_at(curr_tile) > height {
                break;
            }
            if field.is_occupied(curr_tile) {
                self.damage_mob(field, curr_tile, weapon.damage());
                break;
            }
        }
        if weapon.ammo() == &Arrow {
            // arrows can be reused, but sometimes break
            let rng: f32 = rand::random();
            if rng > SETTINGS.read().unwrap().player.arrow_break_chance {
                field.add_loot_at(vec![I(Arrow)], curr_tile);
            } else {
                self.add_message(&"Arrow broke");
            }
            self.animations_buffer.add_projectile_animation(ProjectileType::Arrow, (self.x, self.y), curr_tile);
        }
        self.get_speed_multiplier()
    }

    pub fn shoot_current(&mut self, field: &mut Field) -> f32 {
        self.shoot(field, self.ranged_weapon)
    }

    pub fn load_interactable(&mut self, field: &mut Field,
                             item: &Storable, amount: u32) {
        if self.interacting_with.is_none() {
            self.add_message(&format!("No interactable active"));
            return;
        }
        if self.inventory.drop(item, amount) {
            field.load_interactable_at(self.interacting_with.unwrap(), *item, amount);
        } else {
            self.add_message(&format!("You do not have {} of {}", amount, item));
        }
    }

    pub fn unload_interactable(&mut self, field: &mut Field,
                               item: &Storable, amount: u32) {
        if self.interacting_with.is_none() {
            self.add_message(&format!("No interactable active"));
            return;
        }
        if field.unload_interactable_at(self.interacting_with.unwrap(), item, amount) {
            self.inventory.pickup(*item, amount);
        } else {
            self.add_message(&format!("Interactable does not have {} of {}", amount, item));
        }
    }

    fn damage_mob(&mut self, field: &mut Field, pos: (i32, i32), dmg: i32) {
        let mob_kind = field.get_mob_kind_at(pos).unwrap();
        let died = field.damage_mob(pos, dmg);
        if died {
            self.score_killed_mob(&mob_kind);
        }
    }
    
    /// Rotates the Player 90 degrees (counter-)clockwise.
    ///
    /// # Arguments
    ///
    /// * `side`: -1 or 1; 1 is counterclockwise.
    /// returns: how much action was spent.
    pub fn turn(&mut self, side: i32) -> f32 {
        if self.interacting_with.is_some() {
            self.interacting_with = None;
        }
        self.rotation += side as f32;
        0.25 * self.get_speed_multiplier()
    }

    /// Computes delta_x and delta_y that, added to the Player position, give a cell in front of him.
    pub fn coords_from_rotation(&self) -> (i32, i32) {
        (-(self.rotation * PI / 2.).cos() as i32, -(self.rotation * PI / 2.).sin() as i32)
    }

    /// Computes rotation as an integer between 0 and 4, where 0 is up, and 3 is right.
    pub fn get_rotation(&self) -> u32 {
        let modulus = self.rotation as i32 % 4;
        if modulus >= 0 {
            modulus as u32
        } else {
            (modulus + 4) as u32
        }
    }
    
    pub fn coords_infront(&self) -> (i32, i32) {
        let (delta_x, delta_y) = self.coords_from_rotation();
        (self.x + delta_x, self.y + delta_y)
    }

    pub fn add_status_effect(&mut self, effect: StatusEffect, duration: i32) {
        self.status_effects.push((effect, duration));
    }

    pub fn step_status_effects(&mut self) {
        self.status_effects = self.status_effects.iter().map(|(effect, duration)| {
            (*effect, duration - 1)
        }).filter(|(_, duration)| *duration > 0).collect::<Vec<(StatusEffect, i32)>>();
    }

    pub fn get_speed_multiplier(&self) -> f32 {
        let mut multiplier = 1.;
        for (effect, _) in &self.status_effects {
            match effect {
                StatusEffect::Speedy => {
                    multiplier = 0.5;
                    break
                }
            }
        }
        multiplier
    }

    pub fn get_status_effects(&self) -> &Vec<(StatusEffect, i32)> {
        &self.status_effects
    }

    pub fn has(&self, storable: &Storable) -> bool {
        self.inventory.contains(storable)
    }

    pub fn inventory_count(&self, storable: &Storable) -> u32 {
        self.inventory.count(storable)
    }

    pub fn get_inventory(&self) -> &Vec<(Storable, u32)> {
        self.inventory.get_all()
    }

    pub fn toggle_map(&mut self) {
        self.viewing_map = !self.viewing_map;
    }

    pub fn add_message(&mut self, new: &str) {
        if !self.message.is_empty() {
            self.message.push_str("\n");
        }
        self.message.push_str(new);
    }
    pub fn reset_message(&mut self) {
         self.message = String::new();
     }
}

/// HP and damage things
impl Player {
    pub fn receive_damage(&mut self, damage: i32) {
        self.add_message(&format!("You are hit by {}", damage));
        self.hp -= damage
    }

    pub fn heal(&mut self, hp: i32) {
        self.hp = min(SETTINGS.read().unwrap().player.max_hp, self.hp + hp)
    }

    pub fn get_melee_damage(&self) -> i32 {
        10 + self.inventory.damage_modifier()
    }

    pub fn get_hp(&self) -> i32 {
        self.hp
    }
}
