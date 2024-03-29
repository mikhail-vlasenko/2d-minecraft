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
use crate::graphics::ui::interactables_menu::InteractablesMenu;
use crate::graphics::ui::main_menu::MainMenu;
use crate::map_generation::field::Field;
use crate::map_generation::mobs::mob_kind::MobKind;


/// Renders UI
pub struct EguiManager {
    platform: Platform,
    render_pass: egui_wgpu_backend::RenderPass,
    pub craft_menu_open: RefCell<bool>,
    pub interactables_menu: InteractablesMenu,
    pub main_menu: MainMenu,
    pub main_menu_open: RefCell<bool>,
}

impl EguiManager {
    pub fn new(
        window: &Window,
        size: &PhysicalSize<u32>,
        surface: &Surface,
        adapter: &Adapter,
        device: &Device,
    ) -> Self {
        let surface_format = surface.get_capabilities(&adapter).formats[0];

        let platform = Platform::new(PlatformDescriptor {
            physical_width: size.width as u32,
            physical_height: size.height as u32,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Default::default(),
        });

        let render_pass = egui_wgpu_backend::RenderPass::new(&device, surface_format, 1);

        Self {
            platform,
            render_pass,
            craft_menu_open: RefCell::new(false),
            interactables_menu: InteractablesMenu::new(),
            main_menu: MainMenu::new(),
            main_menu_open: RefCell::new(true),
        }
    }

    /// Renders all of the necessary UI elements.
    pub fn render_ui(&mut self,
                     config: &SurfaceConfiguration,
                     device: &Device,
                     queue: &Queue,
                     encoder: &mut CommandEncoder,
                     view: &TextureView,
                     window: &Window,
                     player: &mut Player,
                     field: &mut Field,
    ) -> TexturesDelta {
        self.platform.begin_frame();

        if *self.main_menu_open.borrow() {
            self.main_menu.render_main_menu(&self.platform, player, config.width as f32 / 2.1);
        } else if !player.viewing_map {
            self.render_place_craft_menu(player);
            self.render_inventory(player);
            self.render_info(player, field.get_time());
            if player.interacting_with.is_some() {
                self.interactables_menu.render_interact_menu(&self.platform, player, field, config.width as f32 / 2.1);
            }
            if *self.craft_menu_open.borrow() {
                self.render_craft_menu(player, config.width as f32 / 2.1);
            }
        }

        if player.get_hp() <= 0 {
            self.render_game_over();
        }

        // End the UI frame. We could now handle the output and draw the UI with the backend.
        let full_output = self.platform.end_frame(Some(window));
        let paint_jobs = self.platform.context().tessellate(full_output.shapes);

        // Upload all resources for the GPU.
        let screen_descriptor = ScreenDescriptor {
            physical_width: config.width,
            physical_height: config.height,
            scale_factor: window.scale_factor() as f32,
        };

        let texture_delta: TexturesDelta = full_output.textures_delta;
        self.render_pass
            .add_textures(&device, &queue, &texture_delta)
            .expect("add texture ok");
        self.render_pass.update_buffers(&device, &queue, &paint_jobs, &screen_descriptor);

        // Record all render passes.
        self.render_pass
            .execute(
                encoder,
                &view,
                &paint_jobs,
                &screen_descriptor,
                None,
            )
            .unwrap();

        texture_delta
    }

    pub fn post_render_cleanup(&mut self, texture_delta: TexturesDelta) {
        self.render_pass
            .remove_textures(texture_delta)
            .expect("remove texture ok");
    }

    fn render_place_craft_menu(&self, player: &mut Player) {
        egui::Window::new("Menu").anchor(Align2::LEFT_TOP, [0., 0.]).auto_sized()
            .show(&self.platform.context(), |ui| {
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
    fn render_craft_menu(&self, player: &mut Player, width: f32) {
        egui::Window::new("Craft Menu").anchor(Align2::CENTER_CENTER, [0., 0.])
            .collapsible(false)
            .fixed_size([width, 300.])
            .show(&self.platform.context(), |ui| {
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

    fn render_inventory(&self, player: &Player) {
        egui::Window::new("Inventory").anchor(Align2::LEFT_BOTTOM, [0., 0.])
            .show(&self.platform.context(), |ui| {
                for item in player.get_inventory() {
                    if item.1 != 0 {
                        ui.label(format!("{}: {}", item.0, item.1));
                    }
                }
            });
    }

    fn render_info(&self, player: &Player, time: f32) {
        egui::Window::new("Info").anchor(Align2::RIGHT_TOP, [0., 0.])
            .show(&self.platform.context(), |ui| {
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
                ui.label(format!("{}", player.message));
            });
    }

    fn render_game_over(&self) {
        egui::Window::new("").anchor(Align2::CENTER_CENTER, [0., 0.])
            .show(&self.platform.context(), |ui| {
                ui.add(Label::new(RichText::new("Game Over!")
                    .font(FontId::proportional(80.0))
                    .color(Color32::RED)
                    .strong()
                ).wrap(false));
            });
    }

    pub fn handle_event<T>(&mut self, event: &Event<T>) {
        self.platform.handle_event(event);
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

