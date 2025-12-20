use std::{fs, io};
use std::error::Error;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use rand::Rng;
use crate::auxiliary::replay::Replay;
use crate::character::milestones::MilestoneTracker;
use crate::character::player::Player;
use crate::map_generation::field::Field;
use crate::SETTINGS;


trait BinaryString {
    fn to_binary_string(&self) -> Vec<u8>;
    fn from_binary_string(data: &[u8]) -> Result<Self, postcard::Error> where Self: Sized;
}

impl BinaryString for Field {
    fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    fn from_binary_string(data: &[u8]) -> Result<Self, postcard::Error> {
        let mut deserialized: Field = postcard::from_bytes(data)?;
        let central_chunk = deserialized.get_central_chunk();
        deserialized.load(central_chunk.0, central_chunk.1);
        Ok(deserialized)
    }
}

impl BinaryString for Player {
    fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    fn from_binary_string(data: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(data)
    }
}

impl BinaryString for MilestoneTracker {
    fn to_binary_string(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap()
    }

    fn from_binary_string(data: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(data)
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

pub fn load_game(path: &Path) -> Result<(Field, Player, Replay, MilestoneTracker), Box<dyn Error>> {
    fn read_file(path: &Path) -> Result<Vec<u8>, io::Error> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(data)
    }

    let field = Field::from_binary_string(&read_file(&path.join("field.postcard"))?)?;
    let player = Player::from_binary_string(&read_file(&path.join("player.postcard"))?)?;
    let milestone_tracker = match read_file(&path.join("milestone_tracker.postcard")) {
        Ok(data) => MilestoneTracker::from_binary_string(&data).unwrap_or_else(|_| MilestoneTracker::new()),
        Err(_) => MilestoneTracker::new(),
    };

    Ok((field, player, Replay::new(), milestone_tracker))
}

/// Returns a list of directories or files in the given path.
/// Name and seconds since epoch of last modification are returned.
/// 
/// # Arguments
/// 
/// * `path` - The path to list directories or files in.
/// * `directories` - If true, directories are listed. If false, files are listed.
pub fn list_directory(path: &Path, directories: bool) -> Result<Vec<(String, i32)>, Box<dyn Error>> {
    let mut result = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        // If the entry is a directory, add it to the list
        if (directories && path.is_dir()) || (!directories && path.is_file()) {
            if let Some(filename) = path.file_name() {
                let metadata = fs::metadata(&path)?;
                let modified = metadata.modified()?.duration_since(std::time::SystemTime::UNIX_EPOCH)?.as_secs() as i32;
                result.push((filename.to_string_lossy().into_owned(), modified));
            }
        }
    }

    Ok(result)
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

    let suffix: String = rand::thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(4)
        .map(char::from)
        .collect();
    name.push_str(&format!("_{}", suffix));

    let mut path = PathBuf::from(SETTINGS.read().unwrap().save_folder.clone().into_owned());
    path.push(&name);
    save_game(field, player, milestone_tracker, &path);
    name
}
