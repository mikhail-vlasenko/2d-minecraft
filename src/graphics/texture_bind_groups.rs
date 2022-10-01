use wgpu::{BindGroup, BindGroupLayout};
use crate::graphics::state::State;
use crate::graphics::texture::Texture;


pub struct TextureBindGroups {
    pub grass: BindGroup,
    pub stone: BindGroup,
    pub tree_log: BindGroup,
    pub bedrock: BindGroup,
    pub planks: BindGroup,
    pub player: BindGroup,
}

impl TextureBindGroups{
    fn make_bind_group(
        label: &str, texture: &Texture, device: &wgpu::Device, texture_bind_group_layout: &BindGroupLayout
    ) -> BindGroup {
        device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture.sampler),
                    }
                ],
                label: Some(label),
            }
        )
    }

    pub fn init_bind_groups(
        device: &wgpu::Device, queue: &wgpu::Queue, texture_bind_group_layout: &BindGroupLayout
    ) -> TextureBindGroups {
        let grass_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mc_grass.png"), "mc_grass.png"
        ).unwrap();
        let grass = Self::make_bind_group(
            "grass_bind_group", &grass_texture, &device, &texture_bind_group_layout
        );

        let stone_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mc_stone.png"), "mc_stone.png"
        ).unwrap();
        let stone = Self::make_bind_group(
            "stone_bind_group", &stone_texture, &device, &texture_bind_group_layout
        );

        let tree_log_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mc_tree_log.png"), "mc_tree_log.png"
        ).unwrap();
        let tree_log = Self::make_bind_group(
            "tree_log_bind_group", &tree_log_texture, &device, &texture_bind_group_layout
        );

        let bedrock_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mc_bedrock.png"), "bedrock.png"
        ).unwrap();
        let bedrock = Self::make_bind_group(
            "bedrock_bind_group", &bedrock_texture, &device, &texture_bind_group_layout
        );

        let planks_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mc_planks.png"), "planks.png"
        ).unwrap();
        let planks = Self::make_bind_group(
            "bedrock_bind_group", &planks_texture, &device, &texture_bind_group_layout
        );

        let player_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/player_top_view.png"), "player.png"
        ).unwrap();
        let player = Self::make_bind_group(
            "player_bind_group", &player_texture, &device, &texture_bind_group_layout
        );

        TextureBindGroups {
            grass,
            stone,
            tree_log,
            bedrock,
            planks,
            player,
        }
    }
}
