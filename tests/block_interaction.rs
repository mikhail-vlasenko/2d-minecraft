use minecraft::crafting::items::Item;
use minecraft::crafting::material::Material;
use minecraft::crafting::storable::Storable;
use minecraft::map_generation::chunk::Chunk;
use minecraft::map_generation::field::Field;
use minecraft::map_generation::read_chunk::read_file;
use minecraft::player::Player;

#[test]
fn test_player_basic() {
    let field = Field::new(None);
    let player = Player::new(&field);

    assert_eq!(player.get_hp(), 100);
    assert_eq!(player.get_mining_power(), 0);
    assert_eq!(player.get_melee_damage(), 10);
}

#[test]
fn test_player_good_weather() {
    let test_chunk = Chunk::from(read_file(String::from("res/chunks/test_chunk.txt")));

    // Mobs dont step here, and the 0, 0 chunk doesnt spawn a mob
    let mut field = Field::new(Some(test_chunk));
    let mut player = Player::new(&field);

    player.turn(-1);
    player.mine_infront(&mut field);
    player.mine_infront(&mut field);
    player.walk("s", &mut field);
    player.walk("d", &mut field);
    player.mine_infront(&mut field);
    player.mine_infront(&mut field);

    player.crafting_item = Storable::M(Material::Plank);
    player.craft_current(&field);
    player.craft_current(&field);
    player.craft_current(&field);
    player.crafting_item = Storable::M(Material::CraftTable);
    player.craft_current(&field);
    player.placement_material = Material::CraftTable;
    player.place_current(&mut field);
    player.crafting_item = Storable::I(Item::Stick);
    player.craft_current(&field);
    player.crafting_item = Storable::I(Item::WoodenPickaxe);
    assert_eq!(player.get_mining_power(), 0);
    player.craft_current(&field);

    assert_eq!(player.get_mining_power(), 1);
}

#[test]
fn test_player_no_crafttable() {
    let test_chunk = Chunk::from(read_file(String::from("res/chunks/test_chunk.txt")));

    // Mobs dont step here, and the 0, 0 chunk doesnt spawn a mob
    let mut field = Field::new(Some(test_chunk));
    let mut player = Player::new(&field);

    player.turn(-1);
    player.mine_infront(&mut field);
    player.mine_infront(&mut field);
    player.walk("s", &mut field);
    player.walk("d", &mut field);
    player.mine_infront(&mut field);
    player.mine_infront(&mut field);

    player.crafting_item = Storable::M(Material::Plank);
    player.craft_current(&field);
    player.craft_current(&field);
    player.craft_current(&field);
    player.crafting_item = Storable::M(Material::CraftTable);
    player.craft_current(&field);
    player.crafting_item = Storable::I(Item::Stick);
    player.craft_current(&field);
    player.crafting_item = Storable::I(Item::WoodenPickaxe);
    player.craft_current(&field);

    assert_eq!(player.get_mining_power(), 0);
}
