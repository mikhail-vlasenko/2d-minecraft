use minecraft::map_generation::field::Field;
use minecraft::player::Player;

#[test]
fn test_player_basic() {
    let field = Field::new();
    let player = Player::new(&field);

    assert_eq!(player.get_hp(), 100);
    assert_eq!(player.get_mining_power(), 0);
    assert_eq!(player.get_melee_damage(), 10);
}