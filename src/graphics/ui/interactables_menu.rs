use std::cell::RefCell;
use egui::{Align, Checkbox, FontDefinitions, Slider};
use egui::{Align2, Color32, FontId, Label, RichText, TexturesDelta};
use egui_wgpu_backend;
use egui_wgpu_backend::ScreenDescriptor;
use egui_winit_platform::{Platform, PlatformDescriptor};
use wgpu::{Adapter, CommandEncoder, Device, Queue, Surface, SurfaceConfiguration, TextureView};
use winit::dpi::PhysicalSize;
use winit::event::Event;
use winit::window::Window;
use crate::character::player::Player;
use strum::IntoEnumIterator;
use crate::crafting::consumable::Consumable;
use crate::crafting::interactable::{Interactable, InteractableKind};
use crate::crafting::items::Item;
use crate::crafting::material::Material;
use crate::crafting::ranged_weapon::RangedWeapon;
use crate::crafting::storable::{CraftMenuSection, Storable};
use crate::crafting::storable::Craftable;
use crate::map_generation::field::Field;
use crate::map_generation::mobs::mob_kind::MobKind;


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
        platform: &Platform,
        player: &mut Player,
        field: &mut Field,
        width: f32,
    ) {
        let inter_pos = player.interacting_with.unwrap();
        let kind = field.get_interactable_kind_at(inter_pos).unwrap();
        egui::Window::new(format!("{} menu", kind))
            .anchor(Align2::CENTER_CENTER, [0., 0.])
            .collapsible(false)
            .fixed_size([width, 300.])
            .show(&platform.context(), |ui| {
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