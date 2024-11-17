use std::{fs, io};
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use crate::character::milestones::MilestoneTracker;
use crate::character::player::Player;
use crate::map_generation::field::Field;
use crate::SETTINGS;


trait BinaryString {
    fn to_binary_string(&self) -> Vec<u8>;
    fn from_binary_string(data: &Vec<u8>) -> Self;
}

impl BinaryString for Field {
    fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    fn from_binary_string(data: &Vec<u8>) -> Self {
        let mut deserialized: Field = postcard::from_bytes(data).unwrap();
        let central_chunk = deserialized.get_central_chunk();
        deserialized.load(central_chunk.0, central_chunk.1);
        deserialized
    }
}

impl BinaryString for Player {
    fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    fn from_binary_string(data: &Vec<u8>) -> Self {
        postcard::from_bytes(data).unwrap()
    }
}

impl BinaryString for MilestoneTracker {
    fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    fn from_binary_string(data: &Vec<u8>) -> Self {
        postcard::from_bytes(data).unwrap()
    }
}

pub fn save_game(field: &Field, player: &Player, milestone_tracker: &MilestoneTracker, path: &Path) {
    create_dir_all(&path).unwrap();

    let save_data = [
        ("field.postcard", field.to_binary_string()),
        ("player.postcard", player.to_binary_string()),
        ("milestone_tracker.postcard", milestone_tracker.to_binary_string()),
    ];

    for (filename, serialized_data) in save_data {
        let file_path = path.join(filename);
        let mut file = File::create(file_path).unwrap();
        file.write_all(&serialized_data).unwrap();
    }
}

pub fn load_game(path: &Path) -> (Field, Player, MilestoneTracker) {
    fn read_file(path: &Path) -> Vec<u8> {
        let mut file = File::open(path).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        data
    }

    let field = Field::from_binary_string(&read_file(&path.join("field.postcard")));
    let player = Player::from_binary_string(&read_file(&path.join("player.postcard")));
    let milestone_tracker = File::open(path.join("milestone_tracker.postcard"))
        .map_or_else(
            |_| MilestoneTracker::new(),
            |mut file| {
                let mut data = Vec::new();
                file.read_to_end(&mut data)
                    .map(|_| MilestoneTracker::from_binary_string(&data))
                    .unwrap_or_else(|_| MilestoneTracker::new())
            }
        );
    
    (field, player, milestone_tracker)
}

pub fn get_directories(path: &Path) -> io::Result<Vec<String>> {
    let mut directories = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        // If the entry is a directory, add it to the list
        if path.is_dir() {
            if let Some(filename) = path.file_name() {
                directories.push(filename.to_string_lossy().into_owned());
            }
        }
    }
    directories.sort();

    Ok(directories)
}

pub fn get_files(path: &Path) -> io::Result<Vec<String>> {
    let mut files = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        // If the entry is a file, add it to the list
        if path.is_file() {
            if let Some(filename) = path.file_name() {
                files.push(filename.to_string_lossy().into_owned());
            }
        }
    }
    files.sort();

    Ok(files)
}

pub fn get_full_path(path: &Path) -> PathBuf {
    let working_dir = std::env::current_dir().unwrap_or(PathBuf::new());
    working_dir.join(path)
}

pub fn autosave_game(field: &Field, player: &Player, milestone_tracker: &MilestoneTracker) -> String {
    let mut name = String::from("autosave_");
    name.push_str(&format!("ms_{}_", milestone_tracker.get_current_milestone_idx()));
    name.push_str(&format!("score_{}_", player.get_score()));
    name.push_str(&chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string());
    let mut path = PathBuf::from(SETTINGS.read().unwrap().save_folder.clone().into_owned());
    path.push(&name);
    save_game(field, player, milestone_tracker, &path);
    name
}
