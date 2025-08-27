use serde::{Deserialize, Serialize};
use crate::character::player::Player;
use crate::character::status_effects::StatusEffect;
use crate::map_generation::field::AbsolutePos;


#[derive(PartialEq, Clone, Hash, Serialize, Deserialize, Debug)]
pub enum P2PInteraction {
    /// The AbsolutePos is the position of the target player
    DealDamage(i32),
    Heal(i32),
    ApplyStatusEffect(StatusEffect, i32),  // effect and duration in turns
    IncreaseScore(i32), // amount to increase the score by
    AddMessage(String), // message to add to the player's log
}

pub type InteractionStore = Vec<(P2PInteraction, AbsolutePos)>;

/// Applies a list of interactions to the players.
///
/// # Arguments
/// * `interactions` - A vector of tuples where each tuple contains a P2PInteraction and the AbsolutePos of the target player.
/// * `players` - A mutable reference to a vector of players all in the game.
///
pub fn apply_p2p_interactions(interactions: InteractionStore, players: Vec<Player>) -> Vec<Player> {
    let mut player_map = std::collections::HashMap::new();
    let original_len = players.len();
    for player in players {
        player_map.insert(player.xy(), player);
    }
    assert_eq!(player_map.len(), original_len, "There are players with duplicate positions! This should never happen.");
    for (interaction, target_pos) in interactions {
        let target_player = player_map.get_mut(&target_pos);
        if target_player.is_none() {
            println!("Position {:?} has no player! Perhaps they died recently.", target_pos);
            continue;
        }
        let target_player = target_player.unwrap();
        use P2PInteraction::*;
        match interaction {
            DealDamage(dmg) => target_player.receive_damage(dmg),
            Heal(heal) => target_player.heal(heal),
            ApplyStatusEffect(effect, duration) => target_player.add_status_effect(effect, duration),
            IncreaseScore(amount) => target_player.add_to_score(amount),
            AddMessage(msg) => target_player.add_message(&msg),
        }
    }
    player_map.into_values().collect()
}
