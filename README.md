# Top-down view Minecraft

This is my clone of Minecraft, built without a game engine, purely on WGPU (and some egui for interface).

It is a turn-based, 2.5d game: the field looks 2d, but you can actually break the top block, or place a block on top of an existing one.

The grey circle indicates the height of the top block (the one you see). The darker the circle, the lower the block.
<img width="51" alt="image" src="https://user-images.githubusercontent.com/27450370/198854409-9959c002-301e-4873-b9f6-7f2131c82fad.png">

Key features:
 - The game has crafting, and for some items you will need a crafting table nearby.
 - The map is infinite in 2 dimensions.
 - Your attack damage depends on the items in your inventory.
 - You need different tiers of pickaxes to break some blocks.
 - Zombies on unloaded chunks do not vanish. They remain in place until the chunk is loaded again.
 - Banelings (hostile mobs) are better than creepers: they blow up and destroy walls if they can't find a route to player, even if the player is a few blocks away from the explosion radius.
 - Zerglings are smart and only attack in groups.
 - Shot arrows can be reused if they don't break.
 - Self-shooting turrets can be placed to help you defend.
 

### AI-friendly
Part of the purpose of this project is to build a training environment for ML agents.

Such evironment would be similar to the famous [MineRL](https://minerl.io/), as it could have the same complexity in terms of progression: 
mine tree -> make crafting table -> make pickaxe -> etc. 

But would require no computer vision, and would run significantly faster than the original.

## Gameplay screenshots:
<img width="801" alt="image" src="https://user-images.githubusercontent.com/27450370/222000895-111b83c9-b2c2-43f3-88f0-093e4267ed13.png">

<img width="800" alt="image" src="https://github.com/mikhail-vlasenko/2d-minecraft/assets/27450370/2e5c9bd4-b2c1-4944-b81e-4e38d4375851">

<img width="788" alt="image" src="https://user-images.githubusercontent.com/27450370/222001819-a4cee2c1-49e7-4a2a-b093-390b68a0661b.png">

<img width="802" alt="image" src="https://user-images.githubusercontent.com/27450370/208272156-e5e712cf-3fe6-4f8d-afff-71f8c5db42aa.png">

<img width="799" alt="image" src="https://github.com/mikhail-vlasenko/2d-minecraft/assets/27450370/d8c6774c-51b9-4a3c-89f3-cb9470907464">

## Asset Credits

- Most of the blocks - minecraft
- PLayer - Riley Gombart
- Zergling and Baneling - Starcraft 2
- Cow - Minecraft
- Hit animations - Sinestesia
