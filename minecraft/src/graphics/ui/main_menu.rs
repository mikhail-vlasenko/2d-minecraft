use std::path::Path;
use std::process::exit;
use egui::{Slider};
use egui::{Align2, Color32, FontId, RichText, Context};
use game_logic::character::player::Player;
use game_logic::map_generation::save_load::get_directories;
use game_logic::SETTINGS;
use game_logic::settings::Settings;

pub struct MainMenu {
    /// Used to communicate with the main game loop.
    pub selected_option: SelectedOption,
    /// The second panel is the panel that appears on the right side of the main menu.
    pub second_panel: SecondPanelState,
    pub save_name: String,
}

impl MainMenu {
    pub fn new() -> Self {
        MainMenu {
            selected_option: SelectedOption::Nothing,
            second_panel: SecondPanelState::About,
            save_name: String::from("default_save"),
        }
    }

    pub fn render_main_menu(
        &mut self,
        context: &Context,
        player: &mut Player,
        width: f32,
    ) {
        let mut settings = SETTINGS.write().unwrap();  // lock SETTINGS once at the beginning
        egui::Window::new("Main Menu")
            .anchor(Align2::CENTER_CENTER, [0., 0.])
            .collapsible(false)
            .fixed_size([width, 500.])
            .show(context, |ui| {
                ui.columns(2, |columns| {
                    columns[0].add_space(50.0); // Add space below buttons

                    columns[0].vertical_centered(|ui| {
                        if ui.button(RichText::new("New Game")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            self.selected_option = SelectedOption::NewGame;
                        }
                        if ui.button(RichText::new("Save Game")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            // self.selected_option = SelectedOption::SaveGame;
                            self.second_panel = SecondPanelState::SaveGame;
                        }
                        if ui.button(RichText::new("Load Game")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            self.second_panel = SecondPanelState::LoadGame;
                        }
                        if ui.button(RichText::new("Settings")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            self.second_panel = SecondPanelState::Settings;
                        }
                        if ui.button(RichText::new("Controls")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            self.second_panel = SecondPanelState::Controls;
                        }
                        if ui.button(RichText::new("Exit Game")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            exit(0);
                        }
                    });

                    columns[0].add_space(50.0); // Add space below buttons

                    self.second_panel_label(&mut columns[1]);

                    match self.second_panel {
                        SecondPanelState::SaveGame => {
                            columns[1].horizontal(|ui| {
                                ui.label("Save name:");
                                ui.text_edit_singleline(&mut self.save_name);
                            });
                            if columns[1].button("Save").clicked() {
                                self.selected_option = SelectedOption::SaveGame;
                            }
                            self.back_button(&mut columns[1]);
                        }
                        SecondPanelState::LoadGame => {
                            let path_string = settings.save_folder.clone().into_owned();
                            let path = Path::new(&path_string);
                            let save_names = get_directories(&path).unwrap_or(vec![]);

                            // todo: make into a scroll area
                            for name in save_names.iter() {
                                if columns[1].button(RichText::new(name)
                                    .font(FontId::proportional(20.0))).clicked() {
                                    self.save_name = name.clone();
                                    self.selected_option = SelectedOption::LoadGame;
                                }
                            }

                            self.back_button(&mut columns[1]);
                        }
                        SecondPanelState::Settings => {
                            Self::render_difficulty_buttons(&mut columns[1], &mut settings);
                            columns[1].add_space(10.0);
                            Self::render_settings_sliders(&mut columns[1], &mut settings);
                            columns[1].add_space(10.0);

                            let mut save_folder = settings.save_folder.clone().into_owned();
                            columns[1].horizontal(|ui| {
                                ui.label("Save Folder:");
                                ui.text_edit_singleline(&mut save_folder);
                            });
                            settings.save_folder = save_folder.into();
                            self.back_button(&mut columns[1]);
                        }
                        SecondPanelState::Controls => {
                            Self::render_controls(&mut columns[1]);
                            self.back_button(&mut columns[1]);
                        }
                        SecondPanelState::About => {
                            columns[1].label("A short introduction to the game:");
                            columns[1].label("This is a survival game that takes inspiration from Minecraft and Rogue.");
                            columns[1].label("You will have to craft items, fight enemies and explore the world.");
                            columns[1].label("You will be able to build your own base and defend it from enemies.");
                            columns[1].label("Your attack damage depends on the items in your inventory.");
                            columns[1].label("Better pickaxes will allow you to mine more types of blocks.");
                            columns[1].label("Good luck!");
                        }
                    }
                });
            });
    }

    fn render_difficulty_buttons(ui: &mut egui::Ui, settings: &mut Settings) {
        ui.label("Difficulty Preset:");
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Easy").color(Color32::GREEN)).clicked() {
                settings.player.cheating_start = true;
                settings.player.arrow_break_chance = 0.1;
                settings.mobs.spawning.initial_hostile_per_chunk = 0.1;
                settings.mobs.spawning.base_night_amount = 3;
                settings.mobs.spawning.max_mobs_on_chunk = 2;
            }
            if ui.button(RichText::new("Medium").color(Color32::YELLOW)).clicked() {
                settings.player.cheating_start = false;
                settings.player.arrow_break_chance = 0.3;
                settings.mobs.spawning.initial_hostile_per_chunk = 0.2;
                settings.mobs.spawning.base_night_amount = 5;
                settings.mobs.spawning.max_mobs_on_chunk = 3;
            }
            if ui.button(RichText::new("Hard").color(Color32::RED)).clicked() {
                settings.player.cheating_start = false;
                settings.player.arrow_break_chance = 0.5;
                settings.mobs.spawning.initial_hostile_per_chunk = 0.3;
                settings.mobs.spawning.base_night_amount = 7;
                settings.mobs.spawning.max_mobs_on_chunk = 4;
            }
        });
    }

    fn render_settings_sliders(ui: &mut egui::Ui, settings: &mut Settings) {
        ui.label("Mobs Spawning:");
        ui.add(Slider::new(&mut settings.mobs.spawning.initial_hostile_per_chunk, 0.0..=1.0)
            .text("Initial Hostile per Chunk"));
        ui.add(Slider::new(&mut settings.mobs.spawning.base_day_amount, 0..=10)
            .text("Base Day Amount"));
        ui.add(Slider::new(&mut settings.mobs.spawning.base_night_amount, 0..=10)
            .text("Base Night Amount"));
        ui.add(Slider::new(&mut settings.mobs.spawning.increase_amount_every, 1..=10)
            .text("Increase Amount Every (days)"));
        ui.add(Slider::new(&mut settings.mobs.spawning.max_mobs_on_chunk, 0..=10)
            .text("Max Mobs on Chunk"));
        ui.label("Hostile Mob Kind Probabilities (otherwise zombie):");
        ui.add(Slider::new(&mut settings.mobs.spawning.probabilities.bane, 0.0..=1.0)
            .text("Baneling"));
        ui.add(Slider::new(&mut settings.mobs.spawning.probabilities.ling, 0.0..=1.0)
            .text("Zerging"));

        ui.label("Field Generation:");
        ui.add(Slider::new(&mut settings.field.generation.rock_proba, 0.0..=1.0)
            .text("Rock Probability"));
        ui.add(Slider::new(&mut settings.field.generation.tree_proba, 0.0..=1.0)
            .text("Tree Probability"));
        ui.add(Slider::new(&mut settings.field.generation.iron_proba, 0.0..=1.0)
            .text("Iron Probability"));
        ui.add(Slider::new(&mut settings.field.generation.diamond_proba, 0.0..=1.0)
            .text("Diamond Probability"));
        ui.add(Slider::new(&mut settings.field.generation.structures.robot_proba, 0.0..=1.0)
            .text("Robot Probability"));
        ui.checkbox(&mut settings.field.from_test_chunk, "Start with Test Chunk");

        ui.label("Player:");
        ui.checkbox(&mut settings.player.cheating_start, "Cheating Start");
        ui.add(Slider::new(&mut settings.player.arrow_break_chance, 0.0..=1.0)
            .text("Arrow Break Chance"));
    }

    fn render_controls(ui: &mut egui::Ui) {
        let controls = [
            ("WASD", "Move"),
            ("Left Arrow", "Rotate left"),
            ("Right Arrow", "Rotate right"),
            ("Q", "Mine in front"),
            ("E", "Place current item in front"),
            ("C", "Craft current craftable"),
            ("F", "Consume current consumable"),
            ("X", "Shoot current ranged weapon"),
            ("M", "Toggle map view"),
            ("Space", "Toggle craft menu"),
            ("Escape", "Toggle main menu"),
        ];

        for (key, action) in &controls {
            ui.label(format!("{}: {}", key, action));
        }
    }

    fn second_panel_label(&self, ui: &mut egui::Ui) {
        let text = match self.second_panel {
            SecondPanelState::About => "",
            SecondPanelState::SaveGame => "Save Game",
            SecondPanelState::LoadGame => "Load Game",
            SecondPanelState::Settings => "Settings",
            SecondPanelState::Controls => "Controls",
        }.to_string();
        ui.label(RichText::new(text)
            .font(FontId::proportional(20.0))
        );
    }

    fn back_button(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Back").clicked() {
                self.second_panel = SecondPanelState::About;
            }
        });
    }
}

#[derive(PartialEq)]
pub enum SelectedOption {
    Nothing,
    NewGame,
    SaveGame,
    LoadGame,
}

#[derive(PartialEq)]
pub enum SecondPanelState {
    About,
    SaveGame,
    LoadGame,
    Settings,
    Controls,
}
