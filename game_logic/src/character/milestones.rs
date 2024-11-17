use serde::{Deserialize, Serialize};
use crate::character::player::Player;
use crate::crafting::interactable::InteractableKind::CrossbowTurret;
use crate::crafting::items::Item::{DiamondSword, IronIngot, IronPickaxe, WoodenPickaxe};
use crate::crafting::material::Material::{Diamond, IronOre};
use crate::crafting::storable::Storable;


/// Keeps track of the player's progress in the game.
#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct MilestoneTracker {
    milestones: Vec<Milestone>,
    current_milestone: usize,
}

impl MilestoneTracker {
    pub fn new() -> Self {
        let mut milestones = Vec::new();
        milestones.push(Self::crafted_pickaxe());
        milestones.push(Self::mined_3_iron());
        milestones.push(Self::crafted_iron_pickaxe());
        milestones.push(Self::mined_diamond());
        milestones.push(Self::crafted_diamond_sword());
        milestones.push(Self::crafted_crossbow_turret());
        Self {
            milestones,
            current_milestone: 0,
        }
    }
    
    /// Returns true if the current milestone is completed and moves to the next one.
    pub fn check_milestones(&mut self, player: &Player, time: f32) -> bool {
        if self.current_milestone >= self.milestones.len() {
            return false;
        }
        if self.milestones[self.current_milestone].is_completed(player, time) {
            self.current_milestone += 1;
            return true;
        }
        false
    }
    
    /// Milestone index will be 1 after one milestone is completed
    pub fn get_current_milestone_idx(&self) -> usize {
        self.current_milestone
    }
    
    pub fn get_current_milestone_name(&self) -> &String {
        &self.milestones[self.current_milestone].name
    }
    
    fn crafted_pickaxe() -> Milestone {
        Milestone::new(
            "Crafted pickaxe".to_string(), 
            vec![(Storable::I(WoodenPickaxe), 1)], 
            0.
        )
    }
    
    fn mined_3_iron() -> Milestone {
        Milestone::new(
            "Mined 3 iron".to_string(), 
            vec![(Storable::M(IronOre), 3)], 
            0.
        )
    }
    
    fn crafted_iron_pickaxe() -> Milestone {
        Milestone::new(
            "Crafted iron pickaxe".to_string(), 
            vec![(Storable::I(IronPickaxe), 1)], 
            0.
        )
    }
    
    fn mined_diamond() -> Milestone {
        Milestone::new(
            "Diamonds!".to_string(), 
            vec![(Storable::M(Diamond), 1)], 
            0.
        )
    }
    
    fn crafted_diamond_sword() -> Milestone {
        Milestone::new(
            "Diamond sword".to_string(), 
            vec![(Storable::I(DiamondSword), 1)], 
            0.
        )
    }
    
    fn crafted_crossbow_turret() -> Milestone {
        Milestone::new(
            "Crafted crossbow turret".to_string(), 
            vec![(Storable::IN(CrossbowTurret), 1)], 
            0.
        )
    }
}

/// Defines the requirements that need to be met to complete the milestone.
#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
struct Milestone {
    pub name: String,
    /// Must be a subset of the player inventory to complete the milestone.
    inventory_contents: Vec<(Storable, u32)>,
    /// Time must be greater than this to complete the milestone.
    time: f32,
}

impl Milestone {
    pub fn new(name: String, inventory_contents: Vec<(Storable, u32)>, time: f32) -> Self {
        Self {
            name,
            inventory_contents,
            time,
        }
    }
    
    pub fn is_completed(&self, player: &Player, time: f32) -> bool {
        if time < self.time {
            return false;
        }
        for (storable, amount) in &self.inventory_contents {
            // special case to count iron ingots towards iron ore
            if let Storable::M(IronOre) = storable {
                if (player.inventory_count(storable) + player.inventory_count(&Storable::I(IronIngot))) < *amount {
                    return false;
                }
            } else if player.inventory_count(storable) < *amount {
                return false;
            }
        }
        true
    }
}
