use std::cell::RefCell;
use ::egui::FontDefinitions;
use egui::{Align2, Color32, FontId, RichText, TexturesDelta};
use egui_wgpu_backend;
use egui_wgpu_backend::ScreenDescriptor;
use egui_winit_platform::{Platform, PlatformDescriptor};
use wgpu::{Adapter, CommandEncoder, Device, Queue, Surface, SurfaceConfiguration, TextureView};
use winit::dpi::PhysicalSize;
use winit::event::Event;
use winit::window::Window;
use crate::player::Player;
use strum::IntoEnumIterator;
use crate::crafting::consumable::Consumable;
use crate::crafting::items::Item;
use crate::crafting::material::Material;
use crate::crafting::ranged_weapon::RangedWeapon;
use crate::crafting::storable::{CraftMenuSection, Storable};
use crate::crafting::storable::Craftable;


/// Renders UI
pub struct EguiManager {
    platform: Platform,
    render_pass: egui_wgpu_backend::RenderPass,
    pub craft_menu_open: RefCell<bool>,
}

impl EguiManager {
    pub fn new(
        window: &Window,
        size: &PhysicalSize<u32>,
        surface: &Surface,
        adapter: &Adapter,
        device: &Device,
    ) -> Self {
        let surface_format = surface.get_supported_formats(&adapter)[0];

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
        }
    }

    pub fn render_ui(&mut self,
                     config: &SurfaceConfiguration,
                     device: &Device,
                     queue: &Queue,
                     encoder: &mut CommandEncoder,
                     view: &TextureView,
                     window: &Window,
                     player: &mut Player,
                     time: f32,
    ) -> TexturesDelta {
        self.platform.begin_frame();

        self.render_place_craft_menu(player);
        self.render_inventory(player);
        self.render_info(player, time);
        if *self.craft_menu_open.borrow() {
            self.render_craft_menu(player);
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
        egui::Window::new("Menu").show(&self.platform.context(), |ui| {
            ui.label("Placing material");
            for material in Material::iter() {
                if player.has(&Storable::M(material)) {
                    ui.radio_value(&mut player.placement_material, material, material.to_string());
                }
            }

            ui.label(format!("Crafting item: \n{}", player.crafting_item));

            ui.label("Consumable");
            for cons in Consumable::iter() {
                ui.radio_value(
                    &mut player.consumable,
                    cons,
                    format!("{}", cons.to_string()),
                );
            }

            ui.label("Ranged weapon");
            for rw in RangedWeapon::iter() {
                ui.radio_value(
                    &mut player.ranged_weapon,
                    rw,
                    format!("{}", rw.to_string()),
                );
            }
        });
    }

    fn render_craft_menu(&self, player: &mut Player) {
        egui::Window::new("Craft Menu").anchor(Align2::CENTER_CENTER, [0., 0.])
            .collapsible(false)
            .default_width(700.)
            .default_height(300.)
            .show(&self.platform.context(), |ui| {
                let mut menu_iter = CraftMenuSection::iter();
                let count = menu_iter.clone().count();
                ui.columns(count, |columns| {
                    for i in 0..count {
                        let section: CraftMenuSection = menu_iter.next().unwrap();
                        columns[i].label(format!("{:?}", section));
                        for material in Material::iter() {
                            if material.menu_section() == section {
                                let count = player.inventory_count(&material.into());
                                columns[i].selectable_value(
                                    &mut player.crafting_item,
                                    Storable::M(material),
                                    Self::format_item_description(material, count),
                                );
                            }
                        }
                        for item in Item::iter() {
                            if item.menu_section() == section {
                                let count = player.inventory_count(&item.into());
                                columns[i].selectable_value(
                                    &mut player.crafting_item,
                                    Storable::I(item),
                                    Self::format_item_description(item, count),
                                );
                            }
                        }
                        for rw in RangedWeapon::iter() {
                            if rw.menu_section() == section {
                                let count = player.inventory_count(&rw.into());
                                columns[i].selectable_value(
                                    &mut player.crafting_item,
                                    Storable::RW(rw),
                                    Self::format_item_description(rw, count),
                                );
                            }
                        }
                    }
                });
            });
    }

    fn format_item_description(item: impl Craftable, current: u32) -> String {
        if item.is_craftable() {
            format!("{} x{} ({})", item.to_string(), item.craft_yield(), current)
        } else {
            format!("{}", item.to_string())
        }
    }

    fn render_inventory(&self, player: &Player) {
        egui::Window::new("Inventory").show(&self.platform.context(), |ui| {
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
                ui.label(format!("Position: {}, {}, {}", player.x, player.y, player.z));
                ui.label(format!("HP: {}/100", player.get_hp()));
                ui.label(format!("ATK: {}", player.get_melee_damage()));
                ui.label(format!("Mining PWR: {}", player.get_mining_power()));
                ui.label(format!("Time: {}", time));
                ui.label(format!("{}", player.message));
            });
    }

    fn render_game_over(&self) {
        egui::Window::new("").anchor(Align2::CENTER_CENTER, [0., 0.])
            .show(&self.platform.context(), |ui| {
                ui.label(RichText::new("Game Over!")
                    .font(FontId::proportional(80.0))
                    .color(Color32::RED)
                );
            });
    }

    pub fn handle_event<T>(&mut self, event: &Event<T>) {
        self.platform.handle_event(event);
    }
}