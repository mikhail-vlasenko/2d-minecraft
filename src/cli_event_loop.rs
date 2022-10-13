use crate::{Field, Material, Player, Storable};


pub fn cli_event_loop() {
    let mut player = Player::new(25, 25);
    let mut field = Field::new();
    loop {
        field.render(&player);
        player.render_inventory();
        println!("input action type");
        let action: String = read!();
        match action.as_str() {
            "w" => {
                println!("input direction");
                let dir: String = read!();
                player.walk(&*dir, &mut field);
            },
            "m" => {
                println!("input relative coords");
                let x: i32 = read!();
                let y: i32 = read!();
                player.mine(&mut field, x, y)
            },
            "p" => {
                println!("input relative coords and material");
                let x: i32 = read!();
                let y: i32 = read!();
                let material_string: String = read!();
                // only try material, cause we want to place it
                let material = Material::try_from(material_string);
                match material {
                    Err(_) => println!("unrecognized material"),
                    Ok(material) => player.place(&mut field, x, y, material)
                }
            },
            "c" => {
                println!("input the item to craft");
                let item_string: String = read!("{}\n");
                let inferred_item = Storable::try_from(item_string);
                match inferred_item {
                    Err(_) => println!("unknown item (material != item)"),
                    Ok(item) => {
                        if !player.can_craft(&item) {
                            println!("you cannot craft that")
                        } else {
                            player.craft(item);
                            println!("crafting successful")
                        }
                    }
                }
            },
            _ => println!("action not recognized")
        };
    }
}