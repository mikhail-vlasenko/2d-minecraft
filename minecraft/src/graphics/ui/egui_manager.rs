use std::cell::RefCell;
use std::fmt::format;
use egui::{Align2, Color32, FontId, Label, RichText, Context};
use egui_wgpu::ScreenDescriptor;
use egui_wgpu::wgpu::{CommandEncoder, Device, Queue, SurfaceConfiguration, TextureView};
use egui_winit::winit::window::Window;
use game_logic::character::player::Player;
use strum::IntoEnumIterator;
use game_logic::crafting::consumable::Consumable;
use game_logic::crafting::interactable::{InteractableKind};
use game_logic::crafting::items::Item;
use game_logic::crafting::material::Material;
use game_logic::crafting::ranged_weapon::RangedWeapon;
use game_logic::crafting::storable::{CraftMenuSection};
use game_logic::crafting::storable::Craftable;
use crate::graphics::ui::egui_renderer::EguiRenderer;
use crate::graphics::ui::interactables_menu::InteractablesMenu;
use crate::graphics::ui::main_menu::MainMenu;
use game_logic::map_generation::field::Field;


/// Renders UI
pub struct EguiManager {
    pub craft_menu_open: RefCell<bool>,
    pub interactables_menu: InteractablesMenu,
    pub main_menu: MainMenu,
    pub main_menu_open: RefCell<bool>,
    pub save_replay_clicked: bool,
    pub replay_save_name: Option<String>,
}

impl EguiManager {
    pub fn new() -> Self {
        Self {
            craft_menu_open: RefCell::new(false),
            interactables_menu: InteractablesMenu::new(),
            main_menu: MainMenu::new(),
            main_menu_open: RefCell::new(true),
            save_replay_clicked: false,
            replay_save_name: None,
        }
    }

    /// Renders all of the necessary UI elements.
    pub fn render_ui(&mut self,
                     egui_renderer: &mut EguiRenderer,
                     config: &SurfaceConfiguration,
                     device: &Device,
                     queue: &Queue,
                     encoder: &mut CommandEncoder,
                     view: &TextureView,
                     window: &Window,
                     player: &mut Player,
                     field: &mut Field,
    ) {
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [config.width, config.height],
            pixels_per_point: window.scale_factor() as f32,
        };
        egui_renderer.draw(
            device,
            queue,
            encoder,
            window,
            view,
            screen_descriptor,
            |context| {
                if *self.main_menu_open.borrow() {
                    self.main_menu.render_main_menu(context, player, config.width as f32 / 2.1);
                } else if !player.viewing_map {
                    self.render_place_craft_menu(context, player);
                    self.render_inventory(context, player);
                    self.render_info(context, player, field.get_time());
                    if player.interacting_with.is_some() {
                        self.interactables_menu.render_interact_menu(context, player, field, config.width as f32 / 2.1);
                    }
                    if *self.craft_menu_open.borrow() {
                        self.render_craft_menu(context, player, config.width as f32 / 2.1);
                    }
                }

                if player.get_hp() <= 0 {
                    self.render_game_over(context, player);
                }
            },
        )
    }

    fn render_place_craft_menu(&self, context: &Context, player: &mut Player) {
        egui::Window::new("Menu").anchor(Align2::LEFT_TOP, [0., 0.]).auto_sized()
            .show(context, |ui| {
                ui.label("Placing material");
                for material in Material::iter() {
                    let count = player.inventory_count(&material.into());
                    if count > 0 {
                        ui.radio_value(&mut player.placement_storable,
                                       material.into(),
                                       format!("{} ({})", material.to_string(), count));
                    }
                }
                for interactable in InteractableKind::iter() {
                    let count = player.inventory_count(&interactable.into());
                    if count > 0 {
                        ui.radio_value(&mut player.placement_storable,
                                       interactable.into(),
                                       format!("{} ({})", interactable.to_string(), count));
                    }
                }

                ui.label(format!("Crafting item: \n{}", player.crafting_item));

                ui.label("Consumable");
                for cons in Consumable::iter() {
                    let count = player.inventory_count(&cons.into());
                    ui.radio_value(
                        &mut player.consumable,
                        cons,
                        format!("{} ({})", cons.to_string(), count));
                }

                ui.label("Ranged weapon");
                for rw in RangedWeapon::iter() {
                    let count = player.inventory_count(&(*rw.ammo()).into());
                    ui.radio_value(
                        &mut player.ranged_weapon,
                        rw,
                        format!("{} ({} ammo)", rw.to_string(), count),
                    );
                }
            });
    }

    /// Render the menu with all the items for crafting.
    fn render_craft_menu(&self, context: &Context, player: &mut Player, width: f32) {
        egui::Window::new("Craft Menu").anchor(Align2::CENTER_CENTER, [0., 0.])
            .collapsible(false)
            .fixed_size([width, 300.])
            .show(context, |ui| {
                let mut menu_iter = CraftMenuSection::iter();
                let count = menu_iter.clone().count();
                ui.columns(count, |columns| {
                    for i in 0..count {
                        let section: CraftMenuSection = menu_iter.next().unwrap();
                        columns[i].label(format!("{:?}", section));
                        for material in Material::iter() {
                            Self::display_for_section(player, section, columns, material, i);
                        }
                        for item in Item::iter() {
                            Self::display_for_section(player, section, columns, item, i);
                        }
                        for rw in RangedWeapon::iter() {
                            Self::display_for_section(player, section, columns, rw, i);
                        }
                        for it in InteractableKind::iter() {
                            Self::display_for_section(player, section, columns, it, i);
                        }
                    }
                });
            });
    }

    /// Display the item in the given craft menu section.
    fn display_for_section(player: &mut Player,
                           section: CraftMenuSection,
                           columns: &mut [egui::Ui],
                           craftable: impl Craftable,
                           i: usize) {
        if craftable.menu_section() == section {
            let count = player.inventory_count(&craftable.into());
            let has_all = player.has_all_ingredients(&craftable.into());
            columns[i].selectable_value(
                &mut player.crafting_item,
                craftable.into(),
                RichText::new(
                    Self::format_item_description(craftable, count)
                ).color(if has_all || !craftable.is_craftable() {
                    Color32::WHITE
                } else {
                    Color32::LIGHT_RED
                }),
            ).on_hover_text(Self::format_recipe(craftable));
        }
    }

    fn format_item_description(item: impl Craftable, current: u32) -> String {
        if item.is_craftable() {
            format!("{} x{} ({})", item.to_string(), item.craft_yield(), current)
        } else {
            format!("{} ({})", item.to_string(), current)
        }
    }

    fn format_recipe(item: impl Craftable) -> String {
        let mut recipe = String::from("Recipe:");
        for (ingredient, count) in item.craft_requirements() {
            recipe.push_str(&format!("\n{} x{}", ingredient.to_string(), count));
        }
        recipe
    }

    fn render_inventory(&self, context: &Context, player: &Player) {
        egui::Window::new("Inventory").anchor(Align2::LEFT_BOTTOM, [0., 0.])
            .show(context, |ui| {
                for item in player.get_inventory() {
                    if item.1 != 0 {
                        ui.label(format!("{}: {}", item.0, item.1));
                    }
                }
            });
    }

    fn render_info(&self, context: &Context, player: &Player, time: f32) {
        egui::Window::new("Info").anchor(Align2::RIGHT_TOP, [0., 0.])
            .show(context, |ui| {
                ui.add(Label::new(
                    format!("Position: {}, {}, {}", player.x, player.y, player.z)
                ).wrap(false));
                ui.add(Label::new({
                    let mut text = RichText::new(format!("HP: {}/100", player.get_hp()));
                    if player.get_hp() <= 25 {
                        text = text.color(Color32::RED).strong()
                    }
                    text
                }));
                ui.label(format!("ATK: {}", player.get_melee_damage()));
                ui.label(format!("Mining PWR: {}", player.get_mining_power()));
                ui.label(format!("Effects: {:?}", player.get_status_effects()));
                ui.label(format!("Time: {}", time));
                ui.label(time_to_weekday(time));
                ui.label(format!("Score: {}", player.get_score()));
                ui.label(format!("{}", player.message));
            });
    }

    fn render_game_over(&mut self, context: &Context, player: &Player) {
        egui::Window::new("").anchor(Align2::CENTER_CENTER, [0., 0.])
            .show(context, |ui| {
                ui.add(Label::new(RichText::new("Game Over!")
                    .font(FontId::proportional(80.0))
                    .color(Color32::RED)
                    .strong()
                ).wrap(false));
                ui.add(Label::new(RichText::new(format!("Your score: {}", player.get_score()))
                                      .font(FontId::proportional(60.0))
                                      .strong()
                ).wrap(false));
                if ui.button(RichText::new("Save Replay")
                    .font(FontId::proportional(60.0))).clicked() {
                    self.save_replay_clicked = true;
                }
                if self.replay_save_name.is_some() {
                    ui.add(Label::new(RichText::new(
                        format!("Replay saved as: {}", self.replay_save_name.as_ref().unwrap()))
                                          .font(FontId::proportional(20.0))
                    ));
                }
            });
    }
}


fn time_to_weekday(time: f32) -> String {
    let time = time as u32;
    let units_in_day = 100;
    let days_in_week = 7;

    let units_in_week = units_in_day * days_in_week;

    let week = time / units_in_week + 1; // week starts from 1
    let unit_in_week = time % units_in_week;

    let day = unit_in_week / units_in_day;

    let weekday = match day {
        0 => "Mon",
        1 => "Tue",
        2 => "Wed",
        3 => "Thu",
        4 => "Fri",
        5 => "Sat",
        6 => "Sun",
        _ => unreachable!(),
    };

    format!("Week {}, {}", week, weekday)
}

