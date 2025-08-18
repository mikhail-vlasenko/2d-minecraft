use egui::{Align, Slider};
use egui::{Align2, Context};
use game_logic::character::player::Player;
use strum::IntoEnumIterator;
use game_logic::map_generation::field::Field;
use game_logic::map_generation::mobs::mob_kind::MobKind;


pub struct InteractablesMenu {
    pub deposit_ammo_amount: u32,
}

impl InteractablesMenu {
    pub fn new() -> Self {
        InteractablesMenu {
            deposit_ammo_amount: 1,
        }
    }
    pub fn render_interact_menu(
        &mut self,
        context: &Context,
        player: &mut Player,
        field: &mut Field,
        width: f32,
    ) {
        let inter_pos = player.get_interacting_with().unwrap();
        let kind = field.get_interactable_kind_at(inter_pos).unwrap();
        egui::Window::new(format!("{} menu", kind))
            .anchor(Align2::CENTER_CENTER, [0., 0.])
            .collapsible(false)
            .fixed_size([width, 300.])
            .show(context, |ui| {
                let num_columns = 2;
                ui.columns(num_columns, |columns| {
                    columns[0].label(format!("Interactable's inventory:"));
                    for item in field.get_interactable_inventory_at(inter_pos) {
                        if item.1 != 0 {
                            columns[0].label(format!("{}: {}", item.0, item.1));
                        }
                    }
                    if kind.is_turret() {
                        columns[0].add_space(30.);
                        columns[0].label("Turret's targets:");
                        let targets = field.get_interactable_targets_at(inter_pos);
                        let mut new_targets = Vec::new();
                        for mob in MobKind::iter() {
                            let mut mob_in = targets.contains(&mob);
                            columns[0].checkbox(&mut mob_in, format!("{:?}", mob));
                            if mob_in {
                                new_targets.push(mob);
                            }
                        }
                        field.set_interactable_targets_at(inter_pos, new_targets);
                    }
                    columns[0].with_layout(
                        egui::Layout::left_to_right(Align::BOTTOM),
                        |ui| {
                            if ui.button("Take all")
                                .on_hover_text("Take all items from the interactable's inventory.")
                                .clicked() {
                                for (storable, amount) in field.get_interactable_inventory_at(inter_pos) {
                                    player.unload_interactable(field, &storable, amount);
                                }
                            }
                            if kind.is_turret() {
                                let ammo = &kind.get_ammo().unwrap().into();
                                if ui.button(format!("Put {:0>3} ammo", self.deposit_ammo_amount))
                                    .on_hover_text("Put ammo from your inventory into the interactable's inventory.")
                                    .clicked() {
                                    player.load_interactable(
                                        field,
                                        ammo,
                                        self.deposit_ammo_amount,
                                    );
                                    if self.deposit_ammo_amount > player.inventory_count(ammo) {
                                        self.deposit_ammo_amount = player.inventory_count(ammo);
                                    }
                                }
                                ui.add(Slider::new(&mut self.deposit_ammo_amount,
                                                   0..=player.inventory_count(ammo))
                                    .smart_aim(true)
                                    .step_by(1.)
                                    .text("amount"));
                            }
                        });

                    columns[1].label(format!("Your inventory:"));
                    for item in player.get_inventory() {
                        if item.1 != 0 {
                            columns[1].label(format!("{}: {}", item.0, item.1));
                        }
                    }
                });
            });
    }
}