mod common;
use crate::common::Data;

use game_logic::crafting::items::Item::*;
use game_logic::crafting::material::Material::*;
use game_logic::crafting::storable::Storable;
use game_logic::map_generation::field::Field;
use game_logic::character::player::Player;
use game_logic::auxiliary::actions::Action::*;

#[test]
fn test_player_basic() {
    let field = Field::new(8, None);
    let player = Player::new(&field);

    assert_eq!(player.get_hp(), 100);
    assert_eq!(player.get_mining_power(), 0);
    assert_eq!(player.get_melee_damage(), 10);
}

#[test]
fn test_player_good_weather() {
    let mut data = Data::new();

    data.act(TurnRight);
    data.mine();
    data.mine();
    data.act(WalkSouth);
    data.act(WalkEast);
    data.mine();
    data.mine();

    data.craft(Storable::M(Plank));
    data.craft(Storable::M(Plank));
    data.craft(Storable::M(Plank));
    data.craft(Storable::M(CraftTable));
    data.place(CraftTable.into());
    data.craft(Storable::I(Stick));

    assert_eq!(data.player.get_mining_power(), 0);
    data.craft(Storable::I(WoodenPickaxe));
    assert_eq!(data.player.get_mining_power(), 1);
    assert!(data.player.has(&Storable::I(WoodenPickaxe)));

    assert!(!data.player.has(&Storable::I(IronPickaxe)));
    data.act(TurnLeft);
    data.act(TurnLeft);
    data.mine();
    data.mine();
    data.act(TurnLeft);
    data.mine();
    data.mine();
    data.craft(Storable::I(IronIngot));
    data.craft(Storable::I(IronIngot));
    data.craft(Storable::I(IronIngot));
    data.craft(Storable::I(Stick));
    data.craft(Storable::I(IronPickaxe));
    assert!(data.player.has(&Storable::I(IronPickaxe)));
    assert!(!data.player.has(&Storable::I(IronIngot)));
    assert_eq!(data.player.get_mining_power(), 2);

    data.craft(Storable::I(IronSword));
    assert!(!data.player.has(&Storable::I(IronSword)));

    data.act(WalkSouth);
    assert_eq!(data.player.get_position().z, 1);
    data.act(TurnLeft);
    data.mine();
    data.mine();
    data.craft(Storable::I(Stick));

    data.craft(Storable::I(IronSword)); // not enough iron ingots
    assert!(!data.player.has(&Storable::I(IronSword)));

    data.craft(Storable::I(IronIngot));
    data.craft(Storable::I(IronIngot));

    let default_dmg = data.player.get_melee_damage();
    // craft table diagonally
    data.craft(Storable::I(IronSword));
    assert!(data.player.has(&Storable::I(IronSword)));
    assert!(data.player.get_melee_damage() > default_dmg);
}

#[test]
fn test_player_no_craft_table() {
    let mut data = Data::new();

    assert!(!data.player.has(&Storable::I(WoodenPickaxe)));
    data.act(TurnRight);
    data.mine();
    data.mine();
    data.act(WalkSouth);
    data.act(WalkEast);
    data.mine();
    data.mine();

    data.craft(Storable::M(Plank));
    data.craft(Storable::M(Plank));
    data.craft(Storable::M(Plank));
    data.craft(Storable::M(CraftTable));

    data.craft(Storable::I(Stick));

    data.craft(Storable::I(WoodenPickaxe));
    assert_eq!(data.player.get_mining_power(), 0);
    assert!(!data.player.has(&Storable::I(WoodenPickaxe)));

    data.place(CraftTable.into());
    data.act(WalkSouth);
    data.act(WalkSouth);
    data.act(WalkEast);

    // too far from crafting table
    data.craft(Storable::I(WoodenPickaxe));
    assert!(!data.player.has(&Storable::I(WoodenPickaxe)));

    data.act(WalkNorth); // come closer
    data.craft(Storable::I(WoodenPickaxe));
    assert!(data.player.has(&Storable::I(WoodenPickaxe)));
}

#[test]
fn test_player_mining_pwr() {
    let mut data = Data::new();

    data.act(WalkSouth);
    data.act(WalkSouth);
    data.act(WalkSouth);
    data.act(WalkSouth);
    data.act(WalkEast);
    data.act(WalkEast);
    data.act(WalkEast);
    data.mine();
    data.mine();
    data.mine();
    data.mine();
    assert!(!data.player.has(&Storable::M(Stone)));
    data.player.pickup(Storable::I(WoodenPickaxe),1);
    data.mine();
    assert!(data.player.has(&Storable::M(Stone)));
}
